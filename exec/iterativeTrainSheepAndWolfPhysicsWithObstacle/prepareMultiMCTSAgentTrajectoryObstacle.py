import time
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, \
    LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, \
    LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel

class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction):
        self.chooseAction = chooseAction

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicy]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numSimulations, actionSpace, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numSimulations = numSimulations
        self.actionSpace = actionSpace
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy(selfNNModel)
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace, transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue(selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([self.approximatePolicy(NNModel) for NNModel in multiAgentNNModel])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

class CheckAngentStackInWall:
    def __init__(self, wallList,agentMaxSize):
        self.wallList=wallList
        self.agentMaxSize=agentMaxSize
    def __call__(self,qPosList):
        wallCenterList=np.array([wall[:2] for wall in self.wallList])
        wallExpandHalfDiagonalList=np.array([np.add(wall[2:],self.agentMaxSize) for wall in self.wallList])
        posList=qPosList.reshape(-1,2)
        isOverlapList=[np.all(np.abs(np.add(pos,-center))<diag)  for (center,diag) in zip (wallCenterList,wallExpandHalfDiagonalList) for pos in posList]
        return np.any(isOverlapList)

class ResetUniformInEnvWithObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.checkAngentStackInWall=checkAngentStackInWall
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)


        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)

        while self.checkAngentStackInWall(qPos):
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState
def main():

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    iterationIndex = parametersForTrajectoryPath['iterationIndex']

    # parametersForTrajectoryPath={}
    # startSampleIndex=1
    # endSampleIndex=3
    # parametersForTrajectoryPath['sampleIndex']=(startSampleIndex,endSampleIndex)
    # iterationIndex=0

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentObstacle', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    numSimulations = 200
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        # physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgentsTwoObstacles.xml')
        physicsDynamicsPath=os.path.join(dirName,'twoAgentsTwoObstaclesDemo.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        agentMaxSize=0.3
        wallList=[[0,3,0.5,2.8],[0,-3,0.5,2.8]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7

        reset = ResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,checkAngentStackInWall)

        agentIds = list(range(numAgents))
        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus

        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        getApproximatePolicy = lambda NNmodel: ApproximatePolicy(NNmodel, actionSpace)
        getApproximateValue = lambda NNmodel: ApproximateValue(NNmodel)

        getStateFromNode = lambda node: list(node.id.values())[0]

        # neural network init
        numStateSpace = 12
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        trainableAgentIds = [sheepId, wolfId]

        depth = 9
        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for agentId in agentIds]

        startTime = time.time()

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)

        otherAgentApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS)
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpace, terminalRewardList,selectChild, isTerminal, transit, getStateFromNode,getApproximatePolicy, getApproximateValue,composeMultiAgentTransitInSingleAgentMCTS)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy,trainableAgentIds)

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)

        # sample trajectory
        chooseActionList = [chooseActionInMCTS, chooseActionInMCTS]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
