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
    def __init__(self, numSimulations, actionSpace, terminalRewardList, selectChild, isTerminalWithRope, transit, getStateFromNode, getApproximatePolicy, getApproximateValue):
        self.numSimulations = numSimulations
        self.actionSpace = actionSpace
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminalWithRope = isTerminalWithRope
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy[agentId](selfNNModel)
        transitInMCTS = lambda state, selfAction: composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminalWithRope, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminalWithRope, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicies, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicies = approximatePolicies
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([approximatePolicy(NNModel) for approximatePolicy, NNModel in zip(self.approximatePolicies, multiAgentNNModel)])

        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])  for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

class isTerminalWithRope:
    def __init__(self, minXDis, getSheepPos, getWolfPos, getRopeQPos):
        self.minXDis = minXDis
        self.getSheepPos = getSheepPos
        self.getWolfPos = getWolfPos
        self.getRopeQPos = getRopeQPos

    def __call__(self, state):
        state = np.asarray(state)
        posSheep = self.getSheepPos(state)
        posWolf = self.getWolfPos(state)
        posRope = [getPos(state) for getPos in self.getRopeQPos]
        L2NormDistanceForWolf = np.linalg.norm((posSheep - posWolf), ord=2)
        L2NormDistancesForRope = np.array([np.linalg.norm((posSheep - pos), ord=2) for pos in posRope])
        terminal = (L2NormDistanceForWolf <= self.minXDis) or np.any(L2NormDistancesForRope <= self.minXDis)

        return terminal

class RewardFunctionForDistractor():
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

        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        sheepAliveBonus = 0.05
        wolfAlivePenalty = -sheepAliveBonus

        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        masterTerminalPenalty = -1
        distractorTerminalPenalty = 0
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward, masterTerminalPenalty, distractorTerminalPenalty]

        isTerminalWithRope = IsTerminalWithRope(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminalWithRope, numSimulationFrames)

        qPosInit = (0, ) * 26
        qVelInit = (0, ) * 26
        qPosInitNoise = 6
        qVelInitNoise = 5
        numAgent = 4
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(4, 13))
        maxRopePartLength = 0.35

        reset  = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        agentIdForSheepAndDistractorNNState = range(4)
        agentIdForWolfAndMasterNNState = range(3)


# sheep NN model
        sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        sheepPreTrainModel = restoreVariables(initSheepNNModel, sheepPreTrainModelPath)
        sheepPolicy = ApproximatePolicy(sheepPreTrainModel, sheepActionSpace,agentIdForSheepAndDistractorNNState)

# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMasterNNModel, masterPreTrainModelPath)
        masterPolicy = ApproximatePolicy(masterPreTrainModel, masterActionSpace,agentIdForWolfAndMasterNNState)

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace,agentIdForWolfAndMasterNNState)
# distractor NN model
        distractorPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorNNModels','agentId=3_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=100000')
        distractorPreTrainModel = restoreVariables(initdistractorNNModel, distractorPreTrainModelPath)
        distractorPolicy = ApproximatePolicy(distractorPreTrainModel, distractorActionSpace,agentIdForSheepAndDistractorNNState)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preyPowerRatio = 1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.4
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorActionSpace = sheepActionSpace


        actionSpaceList = [sheepActionSpace, wolfActionSpace, masterActionSpace, distractorActionSpace]
        agentIdForNNState = range(3)

        agentIdForSheepAndDistractorNNState = range(4)
        getApproximatePolicy = [lambda NNmodel: ApproximatePolicy(NNmodel, sheepActionSpace,agentIdForSheepAndDistractorNNState), lambda NNmodel: ApproximatePolicy(NNmodel, wolfActionSpace,agentIdForNNState),lambda NNmodel: ApproximatePolicy(NNmodel, masterActionSpace,agentIdForNNState), lambda NNmodel: ApproximatePolicy(NNmodel, distractorActionSpace, agentIdForSheepAndDistractorNNState)]

        safeBound = 2.5
        wallDisToCenter = 10
        wallPunishRatio = 4

### distractor rollout 
        transitInDistractorMCTSSimulation = \
            lambda state, distractorSelfAction: transit(state, [chooseGreedyAction(sheepPolicy(state)), chooseGreedyAction(wolfPolicy(state)), chooseGreedyAction(masterPolicy(state)), distractorSelfAction])
        distractorAliveBonus = 0.05
        distractorDeathPenalty = -1
        velIndex = [4, 5]
        getDistractorVel =  GetAgentPosFromState(distractorId, velIndex)
        velocityBound = 5

        otherIds = list(range(13))
        otherIds.remove(distractorId)
        getOthersPosForDistractor = [GetAgentPosFromState(otherId, qPosIndex) for otherId in otherIds]

        isCollided = IsCollided(killzoneRadius, getDistractorQPos, getOthersPosForDistractor)
        distractorRewardFunction = RewardFunctionForDistractor(distractorAliveBonus, distractorDeathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isCollided, getDistractorQPos, getDistractorVel)

        distractorRolloutPolicy = lambda state: distractorActionSpace[np.random.choice(range(numActionSpace))]
        distractorRolloutHeuristicWeight = 0
        distractorMaxRolloutSteps = 7
        distractorRolloutHeuristic = HeuristicDistanceToTarget(distractorRolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        distractorRollout = RollOut(distractorRolloutPolicy, distractorMaxRolloutSteps, transitInDistractorMCTSSimulation, distractorRewardFunction, isTerminalWithRope, distractorRolloutHeuristic)

#### sheep rollout 
        transitInSheepMCTSSimulation = \
            lambda state, sheepSelfAction: transit(state, [sheepSelfAction, chooseGreedyAction(wolfPolicy(state)), chooseGreedyAction(masterPolicy(state)),chooseGreedyAction(distractorPolicy(state))])

        sheepAliveBonus = 0.05
        sheepDeathPenalty = -1
        sheepRewardFunction = RewardFunctionCompete(sheepAliveBonus, sheepDeathPenalty, isTerminalWithRope)

        sheepRolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
        sheepRolloutHeuristicWeight = sheepDeathPenalty/10
        sheepMaxRolloutSteps = 7
        sheepRolloutHeuristic = HeuristicDistanceToTarget(sheepRolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        sheepRollout = RollOut(sheepRolloutPolicy, sheepMaxRolloutSteps, transitInSheepMCTSSimulation, sheepRewardFunction, isTerminalWithRope,
                          sheepRolloutHeuristic)

#### wolf rollout 
        transitInWolfMCTSSimulation = \
            lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(sheepPolicy(state)), wolfSelfAction, chooseGreedyAction(masterPolicy(state)),chooseGreedyAction(distractorPolicy(state))])

        wolfAliveBonus = -0.05
        wolfDeathPenalty = 1
        wolfRewardFunction = RewardFunctionCompete(wolfAliveBonus, wolfDeathPenalty, isTerminalWithRope)

        wolfRolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]
        wolfRolloutHeuristicWeight = wolfDeathPenalty/10
        wolfMaxRolloutSteps = 7
        wolfRolloutHeuristic = HeuristicDistanceToTarget(wolfRolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        wolfRollout = RollOut(wolfRolloutPolicy, wolfMaxRolloutSteps, transitInWolfMCTSSimulation, wolfRewardFunction, isTerminalWithRope, wolfRolloutHeuristic)

        masterRollout = lambda node: 0
        getApproximateValue = [sheepRollout, wolfRollout,masterRollout, distractorRollout]


        getStateFromNode = lambda node: list(node.id.values())[0]
        # sample trajectory
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminalWithRope, reset, chooseGreedyAction)



# neural network init
        wolfAndMasterNumStateSpace = 18
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(wolfAndMasterNumStateSpace, numActionSpace, regularizationFactor)

        initWolfNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initMasterNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

        distractorNumStateSpace = 24
        generateDistractorModel = GenerateModel(distractorNumStateSpace, numActionSpace, regularizationFactor)
        initdistractorNNModel = generateDistractorModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

        sheepNumStateSpace = 24
        generateSheepModel = GenerateModel(sheepNumStateSpace, numActionSpace, regularizationFactor)
        initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)




        depth = 4



        multiAgentNNmodel = [sheepPreTrainModel, wolfPreTrainModel,masterPreTrainModel, distractorPreTrainModel]

        trainableAgentIds = [sheepId, distractorId]

        startTime = time.time()

        agentIdForSheepAndDistractorNNState = range(4)
        otherAgentApproximatePolicies = [lambda NNmodel: ApproximatePolicy(NNmodel, sheepActionSpace,agentIdForSheepAndDistractorNNState), lambda NNmodel: ApproximatePolicy(NNmodel, wolfActionSpace,agentIdForNNState),lambda NNmodel: ApproximatePolicy(NNmodel, masterActionSpace,agentIdForNNState), lambda NNmodel: ApproximatePolicy(NNmodel, distractorActionSpace, agentIdForSheepAndDistractorNNState)]

        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminalWithRope, transit, getStateFromNode, getApproximatePolicy, getApproximateValue)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicies, trainableAgentIds)


        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)

if __name__ == '__main__':
    main()
