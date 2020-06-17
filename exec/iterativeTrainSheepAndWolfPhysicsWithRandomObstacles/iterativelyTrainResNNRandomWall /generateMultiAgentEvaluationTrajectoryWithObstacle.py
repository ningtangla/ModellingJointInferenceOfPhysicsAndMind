import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
from collections import OrderedDict
import pickle
import pandas as pd
import time
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.envMujocoRandomObstacles import SampleObscalesProperty, SetMujocoEnvXmlProperty, changeWallProperty,TransitionFunction,CheckAngentStackInWall,ResetUniformInEnvWithObstacles,getWallList
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist, RollOut
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, \
    GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.NNGuidedMCTS import ApproximatePolicy,ApproximateValue
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards

def sampleAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedAction = actions[selectedIndex]
    return selectedAction
class RestoreNNModel:
    def __init__(self, getModelSavePath, multiAgentNNModel, restoreVariables):
        self.getModelSavePath = getModelSavePath
        self.multiAgentNNModel = multiAgentNNModel
        self.restoreVariables = restoreVariables

    def __call__(self, agentId, iteration):
        modelPath = self.getModelSavePath({'agentId': agentId, 'iterationIndex': iteration})
        restoredNNModel = self.restoreVariables(self.multiAgentNNModel[agentId], modelPath)

        return restoredNNModel


class PreparePolicy:
    def __init__(self, selfApproximatePolicy, otherApproximatePolicy):
        self.selfApproximatePolicy = selfApproximatePolicy
        self.otherApproximatePolicy = otherApproximatePolicy

    def __call__(self, agentId, multiAgentNNModel):
        multiAgentPolicy = [self.otherApproximatePolicy(NNModel) for NNModel in multiAgentNNModel]
        selfNNModel = multiAgentNNModel[agentId]
        multiAgentPolicy[agentId] = self.selfApproximatePolicy(selfNNModel)
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

def composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, transit):
    multiAgentActions = [sampleAction(policy(state)) for policy in othersPolicy]
    multiAgentActions.insert(agentId, selfAction)
    transitInSelfMCTS = transit(state, multiAgentActions)
    return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numSimulations, actionSpace, terminalRewardList, selectChild, isTerminal, transit,
                 getStateFromNode, getApproximatePolicy, getApproximateValue):
        self.numSimulations = numSimulations
        self.actionSpace = actionSpace
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy(selfNNModel)
        transitInMCTS = lambda state, selfAction: composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace, transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue(selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy

class ComposeSingleAgentMCTS():
    def __init__(self, numSimulations, actionSpaces, agentIdsForNNState, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunctions, selectChild, isTerminal, transit, getApproximateActionPrior):
        self.numSimulations = numSimulations
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunctions = rewardFunctions
        self.rolloutHeuristicFunctions = rolloutHeuristicFunctions
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getApproximateActionPrior = getApproximateActionPrior

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel)
        transitInMCTS = lambda state, selfAction: composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaces[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        rolloutPolicy = lambda state: self.actionSpaces[agentId][np.random.choice(range(len(self.actionSpaces[agentId])))]
        rewardFunction = self.rewardFunctions[agentId]
        rolloutHeuristic = self.rolloutHeuristicFunctions[agentId]
        estimateValue = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, rewardFunction, self.isTerminal, rolloutHeuristic)
        MCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand, estimateValue, backup, establishPlainActionDist)

        return MCTSPolicy

class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([self.approximatePolicy(NNModel) for NNModel in multiAgentNNModel])
        otherAgentPolicyForMCTSAgents = np.array(
            [np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])
             for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array(
            [self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
             for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data',
                                       'multiAgentTrain', 'originVersionAddObstacle', 'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 30
    trainNumSimulations = 200
    killzoneRadius = 2
    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, 'twoAgentsTwoObstacles4.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)
        numAgents = 2

        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
        killzoneRadiusInPlay = 2
        isTerminal = IsTerminal(killzoneRadiusInPlay, getSheepXPos, getWolfXPos)

        sheepAliveBonus = 0.05
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]


        numSimulationFrames = 20
        transit = TransitionFunction(numAgents,physicsSimulation , numSimulationFrames,isTerminal)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        # neural network init and save path
        numStateSpace = 20
        regularizationFactor = 1e-4
        sharedWidths = [128, 128, 128, 128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations,
                             'killzoneRadius': killzoneRadius}
        dirName = os.path.dirname(__file__)
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data',
                                            'multiAgentTrain', 'originVersionAddObstacle', 'NNModel')
        NNModelSaveExtension = ''
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
        multiAgentNNmodel = [generateModel(sharedWidths, actionLayerWidths, valueLayerWidths) for agentId in range(numAgents)]

        # functions to get prediction from NN
        restoreNNModel = RestoreNNModel(getNNModelSavePath, multiAgentNNmodel, restoreVariables)

        # function to prepare policy
        selfApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)
        otherApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)
        preparePolicy = PreparePolicy(selfApproximatePolicy, otherApproximatePolicy)

        # generate a set of starting conditions to maintain consistency across all the conditions
        evalQPosInitNoise = 0
        evalQVelInitNoise = 0

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7
        agentMaxSize=0.6
        wallList=[[0,2.5,0.8,1.95],[0,-2.5,0.8,1.95]]
        # checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)
        checkAngentStackInWall=CheckAngentStackInWall(agentMaxSize)
        # getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, evalQPosInitNoise, evalQVelInitNoise)
        reset=ResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,wallList,checkAngentStackInWall)
        # getResetFromQPosInitDummy = lambda qPosInit: ResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,wallList,checkAngentStackInWall)

        # evalNumTrials = 1000
        # generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
        # evalAllQPosInit = [generateInitQPos() for _ in range(evalNumTrials)]
        # evalAllQVelInit = np.random.uniform(-8, 8, (evalNumTrials, 4))
        # getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, evalAllQPosInit[trial], evalAllQVelInit[trial], numAgents, evalQPosInitNoise, evalQVelInitNoise)
        evalMaxRunningSteps = 50
        chooseGreedyActionlist=[chooseGreedyAction,chooseGreedyAction]
        # getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, getResetFromTrial(trial), chooseGreedyActionlist)
        # allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]
        chooseActionList=[chooseGreedyAction,chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, reset, chooseActionList)
        # save evaluation trajectories
        selfIteration = int(parametersForTrajectoryPath['selfIteration'])
        otherIteration = int(parametersForTrajectoryPath['otherIteration'])
        selfId = int(parametersForTrajectoryPath['selfId'])
        multiAgentIterationIndex = [otherIteration] * numAgents
        multiAgentIterationIndex[selfId] = selfIteration

        restoredMultiAgentNNModel = [restoreNNModel(agentId, multiAgentIterationIndex[agentId]) for agentId in range(numAgents)]

        # NNGuidedMCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        getApproximatePolicy = lambda NNmodel: ApproximatePolicy(NNmodel, actionSpace)
        getApproximateValue = lambda NNmodel: ApproximateValue(NNmodel)

        getStateFromNode = lambda node: list(node.id.values())[0]

        otherAgentApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)

        numSimulations = 600
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpace, terminalRewardList,selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue)

        actionSpaceList = [actionSpace,actionSpace]
        agentIdsForNNState = [range(2), range(2)]
        maxRolloutSteps = 10
        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardFunctions = [rewardSheep, rewardWolf]
        rolloutHeuristicWeights =  [0,0]
        rolloutHeuristicFunctions = [HeuristicDistanceToTarget(weight, getWolfXPos, getSheepXPos) for weight in rolloutHeuristicWeights]

        composeSingleAgentMCTS = ComposeSingleAgentMCTS(numSimulations, actionSpaceList, agentIdsForNNState, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunctions, selectChild, isTerminal, transit, getApproximatePolicy)

        trainableAgentIds = [sheepId, wolfId]
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)

        # policy = preparePolicy(selfId, restoredMultiAgentNNModel)
        # policy = prepareMultiAgentPolicy(restoredMultiAgentNNModel)

        sheepPolicy=ApproximatePolicy(restoredMultiAgentNNModel[sheepId], actionSpace)
        wolfPolicy = ApproximatePolicy(restoredMultiAgentNNModel[wolfId], actionSpace)
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        beginTime = time.time()
        # trajectories = [sampleTrajectory(policy) for sampleTrajectory in
                        # allSampleTrajectories[startSampleIndex:endSampleIndex]]

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]                
        processTime = time.time() - beginTime
        saveToPickle(trajectories, trajectorySavePath)
        print(len(trajectories))

if __name__ == '__main__':
    main()
