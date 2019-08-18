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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, RewardFunctionAvoidCollisionAndWall, IsCollided,HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData , restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, StochasticMCTS, backup, establishPlainActionDistFromMultipleTrees, RollOut
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, SelectSoftmaxAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel

class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace, agentStateIdsForNN):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace
        self.agentStateIdsForNN = agentStateIdsForNN

    def __call__(self, state):
        stateNN = state[self.agentStateIdsForNN]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist


class ApproximateValue:
    def __init__(self, policyValueNet,agentStateIdsForNN):
        self.policyValueNet = policyValueNet
        self.agentStateIdsForNN = agentStateIdsForNN

    def __call__(self, state):
        stateNN = state[self.agentStateIdsForNN]
        stateBatch = [np.concatenate(stateNN)]
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})[0][0]
        return valuePrediction


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction, reasonMindFunctions):
        self.chooseAction = chooseAction
        self.reasonMindFunctions = reasonMindFunctions

    def __call__(self,agentId, state, selfAction, othersPolicy, transit):
        reasonMind = list(self.reasonMindFunctions[agentId])
        del reasonMind[agentId]
        othersPolicyInMCTS = [reason(policy) for reason, policy in zip(reasonMind, othersPolicy )]
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicyInMCTS]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS

class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numTrees, numSimulationsPerTree, actionSpaces, agentIdsForNNState, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximateActionPrior, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numTrees = numTrees
        self.numSimulationsPerTree = numSimulationsPerTree
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximateActionPrior = getApproximateActionPrior
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, self.actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = StochasticMCTS(self.numTrees, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return guidedMCTSPolicy

class ComposeSingleAgentMCTS():
    def __init__(self, numTrees, numSimulationsPerTree, actionSpaces, agentIdsForNNState, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunctions, selectChild, isTerminal, transit, getApproximateActionPrior,composeMultiAgentTransitInSingleAgentMCTS):
        self.numTrees = numTrees
        self.numSimulationsPerTree = numSimulationsPerTree
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunctions = rewardFunctions
        self.rolloutHeuristicFunctions = rolloutHeuristicFunctions
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getApproximateActionPrior = getApproximateActionPrior
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, self.actionSpaces[agentId], self.agentIdsForNNState[agentId])
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaces[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        rolloutPolicy = lambda state: self.actionSpaces[agentId][np.random.choice(range(len(self.actionSpaces[agentId])))]
        rewardFunction = self.rewardFunctions[agentId]
        rolloutHeuristic = self.rolloutHeuristicFunctions[agentId]
        estimateValue = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, rewardFunction, self.isTerminal, rolloutHeuristic)
        MCTSPolicy = StochasticMCTS(self.numTrees, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return MCTSPolicy

class PrepareMultiAgentPolicy:
    def __init__(self, MCTSAgentIds, actionSpaces, agentIdsForNNState, composeSingleAgentPolicy, getApproximatePolicy):
        self.MCTSAgentIds = MCTSAgentIds
        self.actionSpaces = actionSpaces
        self.agentIdsForNNState = agentIdsForNNState
        self.composeSingleAgentPolicy = composeSingleAgentPolicy
        self.getApproximatePolicy = getApproximatePolicy

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy= np.array([self.getApproximatePolicy(NNModel, actionSpace, agentStateIdsForNN) for NNModel, actionSpace, agentStateIdsForNN in zip(multiAgentNNModel,
            self.actionSpaces, self.agentIdsForNNState)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])  for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentPolicy(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy



def main():
    #check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo','trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 360
    numSimulations = 400
    killzoneRadius = 0.5

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    masterPowerRatio = float(parametersForTrajectoryPath['masterPowerRatio'])
    beta = float(parametersForTrajectoryPath['beta'])
    numTrials = endSampleIndex - startSampleIndex
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision3Agents.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        draggerBodyIndex = 8

        #set physical model parameter
        physicsSimulation.model.body_mass[[draggerBodyIndex]] = [13]
        physicsSimulation.set_constants()
        physicsSimulation.forward()

        sheepId = 0
        wolfId = 1
        masterId = 2
        distractorId = 3

        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)

        # isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)
        isTerminal = lambda state : False

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        numAgent = 3
        numRopePart = 9
        qPosInit = (0, ) * 2 * (numAgent + numRopePart)
        qVelInit = (0, ) * 2 * (numAgent + numRopePart)
        qPosInitNoise = 6
        qVelInitNoise = 6
        tiedAgentId = [1, 2]
        ropePartIndex = list(range(numAgent, numAgent + numRopePart))
        maxRopePartLength = 0.35

        reset  = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        # sample trajectory
        selectSoftmaxAction = SelectSoftmaxAction(beta)
        chooseActionList = [chooseGreedyAction, chooseGreedyAction, selectSoftmaxAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

# neural network init
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
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

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initMultiAgentNNModels[wolfId], wolfPreTrainModelPath)

# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMultiAgentNNModels[masterId], masterPreTrainModelPath)

# distractor NN model
        distractorPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorAvoidRopeNNModels','agentId=3_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=100000')
        distractorPreTrainModel = restoreVariables(initMultiAgentNNModels[distractorId], distractorPreTrainModelPath)


        multiAgentNNmodel = [sheepPreTrainModel, wolfPreTrainModel,masterPreTrainModel]#, distractorPreTrainModel]


# MCTS compose
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

# multAgent ApproximatePolicyAndActionPrior
        preyPowerRatio = 1.15
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = masterPowerRatio
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorPowerRatio = 1
        distractorActionSpace = list(map(tuple, np.array(actionSpace) * distractorPowerRatio))

        actionSpaceList = [sheepActionSpace, wolfActionSpace, masterActionSpace]#, distractorActionSpace]
        agentStateIdsForNNList = [range(4), range(3), range(3)]#, range(4)]
        getApproximatePolicy = lambda NNmodel, actionSpace, agentStateIdsForNN: ApproximatePolicy(NNmodel, actionSpace, agentStateIdsForNN)
        getApproximateUniformActionPrior = lambda NNModel, actionSpace, agentStateIdsForNN: lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

# multiAgent ApproximateValue
       # sheepAliveBonus = 0.05
       # wolfAlivePenalty = -sheepAliveBonus

       # sheepTerminalPenalty = -1
       # wolfTerminalReward = 1
       # masterTerminalPenalty = -1
       # distractorTerminalPenalty = 0
       # terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward, masterTerminalPenalty, distractorTerminalPenalty]
       # getStateFromNode = lambda node: list(node.id.values())[0]
       # getApproximateValue = lambda NNmodel, agentStateIdsForNN: ApproximateValue(NNmodel, agentStateIdsForNN)

# multiAgent RewardFunctionForRollout
        aliveBonuses = [0.05, -0.05, 0.05]#, 0.05]
        deathPenalties = [-1, 1, -1]#, -1]

        agentIds = list(range(numAgent))
        getSelfQPoses = [GetAgentPosFromState(Id, qPosIndex) for Id in agentIds]
        otherIdsList = [[wolfId] + ropePartIndex, [sheepId], [sheepId]]#, [sheepId, wolfId, masterId] + ropePartIndex]
        getOthersPoses = [[GetAgentPosFromState(otherId, qPosIndex) for otherId in otherIds] for otherIds in otherIdsList]
        velIndex = [4, 5]
        getSelfVels =  [GetAgentPosFromState(Id, velIndex) for Id in agentIds]

        collisionRadius = 1
        isCollidedFunctions = [IsCollided(collisionRadius, getSelfQPos, getOthersPos) for getSelfQPos, getOthersPos in zip(getSelfQPoses, getOthersPoses)]
        safeBoundes = [2.5, 0.1, 0.1]#, 2]
        wallDisToCenter = 10
        wallPunishRatios = [3, 0, 0]#, 4]
        velocityBounds = [0, 0, 0]#, 6]
        rewardFunctions = [RewardFunctionAvoidCollisionAndWall(aliveBonuses[agentId], deathPenalties[agentId], safeBoundes[agentId], wallDisToCenter, \
                wallPunishRatios[agentId], velocityBounds[agentId], isCollidedFunctions[agentId], getSelfQPoses[agentId], getSelfVels[agentId]) \
                for agentId in agentIds]

        rolloutHeuristicWeights = [-0.1, 0.1, -0.1]#, 0]
        rolloutHeuristics = [HeuristicDistanceToTarget(weight, getWolfQPos, getSheepQPos) for weight in rolloutHeuristicWeights]

        numTrees = 4
        numSimulationsPerTree = 100
        maxRolloutSteps = 5

        betaInMCTS = 1
        chooseActionInMCTS = SampleAction(betaInMCTS)
        #chooseActionInMCTS = chooseGreedyAction

        reasonMindList = np.array([[lambda policy: RandomPolicy(actionSpace) for actionSpace in actionSpaceList ] for subjectiveAgentId in range(numAgent)])

        reasonMindList[sheepId][wolfId] = lambda policy: policy
        reasonMindList[wolfId][sheepId] = lambda policy: policy

        # unknowPolicy = lambda policy, actionSpace : RandomPolicy(actionSpace)
        # sheepKnowMind = [lambda policy : policy] * numAgent
        # sheepKnowMind[masterId] = lambda policy: RandomPolicy(masterActionSpace)
        # wolfKnowMind = [lambda policy : policy] * numAgent
        # wolfKnowMind[masterId] = lambda policy: RandomPolicy(masterActionSpace)
        # masterKnowMind = [lambda policy : policy] * numAgent
        # masterKnowMind[sheepId] = lambda policy: RandomPolicy(sheepActionSpace)
        # masterKnowMind[wolfId] = lambda policy: RandomPolicy(wolfActionSpace)
        # knowMind = [sheepKnowMind, wolfKnowMind, masterKnowMind]
        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS, reasonMindList)
        # composeSingleAgentMCTS = ComposeSingleAgentMCTS(numTrees, numSimulationsPerTree, actionSpaceList, agentStateIdsForNNList, maxRolloutSteps, rewardFunctions, rolloutHeuristics, selectChild, isTerminal, transit, getApproximatePolicy, composeMultiAgentTransitInSingleAgentMCTS)
        composeSingleAgentMCTS = ComposeSingleAgentMCTS(numTrees, numSimulationsPerTree, actionSpaceList, agentStateIdsForNNList, maxRolloutSteps, rewardFunctions, rolloutHeuristics, selectChild, isTerminal, transit, getApproximateUniformActionPrior, composeMultiAgentTransitInSingleAgentMCTS)

        trainableAgentIds = [sheepId, wolfId]

        #imageOthersPolicy[masterId] = lambda policy: lambda state : stationaryAgentPolicy(state)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(trainableAgentIds, actionSpaceList, agentStateIdsForNNList, composeSingleAgentMCTS, getApproximatePolicy)

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for _ in range(numTrials)]
        saveToPickle(trajectories, trajectorySavePath)

if __name__ == '__main__':
    main()
