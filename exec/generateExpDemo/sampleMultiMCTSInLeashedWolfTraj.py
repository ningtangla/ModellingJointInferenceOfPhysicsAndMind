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

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniformForLeashed
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData , restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, sampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel

class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace, agentIdForNNState):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace
        self.agentIdForNNState = agentIdForNNState

    def __call__(self, state):
        stateNN = state[self.agentIdForNNState]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist


class ApproximateValue:
    def __init__(self, policyValueNet,agentIdForNNState):
        self.policyValueNet = policyValueNet
        self.agentIdForNNState = agentIdForNNState

    def __call__(self, state):
        stateNN = state[self.agentIdForNNState]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})[0][0]
        return valuePrediction

def composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, transit):
    multiAgentActions = [sampleAction(policy(state)) for policy in othersPolicy]
    multiAgentActions.insert(agentId, selfAction)
    transitInSelfMCTS = transit(state, multiAgentActions)
    return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numSimulations, actionSpaces, agentIdsForNNState, terminalRewardList, selectChild, isTerminalWithRope, transit, getStateFromNode, getApproximateActionPrior, getApproximateValue):
        self.numSimulations = numSimulations
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximateActionPrior = getApproximateActionPrior
        self.getApproximateValue = getApproximateValue

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminalWithRope, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy

class ComposeSingleAgentMCTS():
    def __init__(self, numSimulations, actionSpaces, agentIdsForNNState, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunction, selectChild, isTerminalWithRope, transit,
            getApproximateActionPrior):
        self.numSimulations = numSimulations
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunctions = rewardFunctions
        self.rolloutHeuristicFunctions = rolloutHeuristicFunctions
        self.selectChild = selectChild
        self.isTerminalWithRope = isTerminalWithRope
        self.transit = transit
        self.getApproximateActionPrior = getApproximateActionPrior

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)
 
        rolloutPolicy = lambda state: actionSpaces[agentId][np.random.choice(range(len(actionSpaces[agentId])))]
        rewardFunction = rewardFunctions[agentId]
        rolloutHeuristic = self.rolloutHeuristicFunctions[agentId]
        estimateValue = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, rewardFunction, isTerminal, rolloutHeuristic)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy

class PrepareMultiAgentPolicy:
    def __init__(self, actionSpaces, agentIdsForNNState, composeSingleAgentPolicy, getApproximatePolicy, MCTSAgentIds):
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.composeSingleAgentPolicy = composeSingleAgentPolicy
        self.getApproximatePolicy = getApproximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([self.getApproximatePolicy(NNModel, actionSpace, agentIdForNNState) for NNModel, actionSpace, agentIdForNNState in zip(multiAgentNNModel,
            self.actionSpaces, self.agentIdsForNNState)])

        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])  for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentPolicy(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

class RewardFunctionAvoidCollisionAndWall():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isCollided, getPosition, getVelocity):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.velocityBound = velocityBound
        self.isCollided = isCollided
        self.getPosition = getPosition
        self.getVelocity = getVelocity

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isCollided(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish =  - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0,self.safeBound -  minDisToWall), 2) / np.power(self.safeBound, 2)

        agentVel = self.getVelocity(state)
        velPunish = -np.abs(self.aliveBonus) if np.linalg.norm(agentVel) <= self.velocityBound else 0

        return reward + wallPunish + velPunish


def main():
    #check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo','trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 125
    numSimulations = 200
    killzoneRadius = 1

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        masterId = 2
        distractorId = 3

        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)

        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        qPosInit = (0, ) * 26
        qVelInit = (0, ) * 26
        qPosInitNoise = 6
        qVelInitNoise = 5
        numAgent = 4
        tiedAgentId = [1, 2]
        ropePartIndex = list(range(4, 13))
        maxRopePartLength = 0.35

        reset  = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
        
        # sample trajectory
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

# neural network init
        numStateSpaceList = [24, 18, 18, 24]
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModels = [GenerateModel(numStateSpace, numActionSpace, regularizationFactor) for numStateSpace in numStateSpaceList]
        depth = 4

        initMultiAgentNNModels = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for generateModel in generateModels]
        
# sheep NN model
        sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        sheepPreTrainModel = restoreVariables(initMultiAgentNNModels[sheepId], sheepPreTrainModelPath)
# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMultiAgentNNModels[wolfId], masterPreTrainModelPath)
# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initMasterAgentNNModels[masterId], wolfPreTrainModelPath)
# distractor NN model
        distractorPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorNNModels','agentId=3_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=100000')
        distractorPreTrainModel = restoreVariables(initMultiAgentNNModels[distractorId], distractorPreTrainModelPath)
        
        multiAgentNNmodel = [sheepPreTrainModel, wolfPreTrainModel,masterPreTrainModel, distractorPreTrainModel]


# MCTS compose
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

# multAgent ApproximatePolicyAndActionPrior
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preyPowerRatio = 1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.4
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorActionSpace = sheepActionSpace

        actionSpaceList = [sheepActionSpace, wolfActionSpace, masterActionSpace, distractorActionSpace]
        agentIdForNNStateList = [4, 3, 3, 4]
        getApproximatePolicy = lambda NNmodel, actionSpace, agentIdForNNState: ApproximatePolicy(NNmodel, actionSpace, agentIdForNNState)
        

# multiAgent ApproximateValue 
       # sheepAliveBonus = 0.05
       # wolfAlivePenalty = -sheepAliveBonus

       # sheepTerminalPenalty = -1
       # wolfTerminalReward = 1
       # masterTerminalPenalty = -1
       # distractorTerminalPenalty = 0
       # terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward, masterTerminalPenalty, distractorTerminalPenalty]
       # getStateFromNode = lambda node: list(node.id.values())[0]
       # getApproximateValue = lambda NNmodel, agentIdForNNState: ApproximateValue(NNmodel, agentIdForNNState)
        
# multiAgent RewardFunctionForRollout 
        aliveBonuses = [0.05, -0.05, 0.05, 0.05]
        deathPenalties = [-1, 1, -1, 1]
        
        getSelfQPoses = [GetAgentPosFromState(Id, qPosIndex) for Id in agentIds]
        otherIdsList = [[wolfId] + ropePartIndex, [sheepId], [sheepId], [sheepId, wolfId, masterId] + ropePartIndex] 
        getOthersPoses = [[GetAgentPosFromState(otherId, qPosIndex) for otherId in otherIds] for otherIds in otherIdsList]
        
        velIndex = [4, 5]
        agentIds = list(range(numAgent))
        getVel =  [GetAgentPosFromState(Id, velIndex) for Id in agentIds]
        
        isCollidedFunctions = [IsCollided(killzoneRadius, getSelfQPos, getOthersPos) for getSelfQPos, getOthersPos in zip(getSelfQPoses, getOthersPoses)]
        safeBoundes = [2.5, 0, 0, 2.5]
        wallDisToCenter = 10
        wallPunishRatios = [4, 0, 0, 4]
        velocityBounds = [0, 0, 0, 5]
        rewardFunctions = [RewardFunctionAvoidCollisionAndWall(aliveBonus[agentId], deathPenalties[agentId], safeBoundes[agentId], wallDisToCenter, \
                wallPunishRatios[agentId], velocityBounds[agentId], isCollidedFunctions[agentId], getSelfQPoses[agentId], getSelfVels[agentId]) \
                for agentId in agentIds]

        rolloutHeuristicWeights = [-0.1, 0.1, -0.1, 0]
        rolloutHeuristics = [HeuristicDistanceToTarget(weight, getWolfQPos, getSheepQPos) for weight in rolloutHeuristicWeights]


        #composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpaceList, agentIdForNNStateList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue)
        maxRolloutSteps = 7
        composeSingleAgentMCTS = ComposeSingleAgentMCTS(numSimulations, actionSpaceList, agentIdForNNStateLism, maxRolloutStep, rewardFunction, rolloutHeuristics, \
                selectChild, isTerminal, transit, getApproximatePolicy)
        
        trainableAgentIds = [sheepId, wolfId, distractorId]
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(actionSpaceList, agentIdForNNStateList, composeSingleAgentMCTS, getApproximatePolicy, trainableAgentIds)

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)

if __name__ == '__main__':
    main()
