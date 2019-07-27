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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist, RollOut
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, sampleAction
from exec.parallelComputing import GenerateTrajectoriesParallel

class PrepareMultiAgentNNPolicyWithAgentSelfNNGuidedMCTS:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy

    def __call__(self, agentId, multiAgentNNModel):
        otherAgentNNModel = multiAgentNNModel[:agentId] + multiAgentNNModel[agentId + 1:]
        otherAgentNNPolicy = [self.approximatePolicy(NNModel) for NNModel in otherAgentNNModel]
        selfNNModel = multiAgentNNModel[agentId]
        selfAgentNNGuidedMCTSPolicy = self.composeSingleAgentGuidedMCTS(agentId, selfNNModel, otherAgentNNPolicy)
        multiAgentPolicy = otherAgentNNPolicy.copy()
        multiAgentPolicy.insert(agentId, selfAgentNNGuidedMCTSPolicy)

        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]

        return policy


def main():
    killzoneRadius = 2
    numSimulations = 100
    maxRunningSteps = 20
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'evaluateSupervisedLearning', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])

    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    parametersForTrajectoryPath['agentId'] = agentId
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7
        reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

        sheepId = 0
        wolfId = 1
        agentIds = list(range(numAgents))
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)


        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        alivePenalty = -0.05
        deathBonus = 1
        rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)


        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

        wolfActionInSheepSimulation = lambda state: (0, 0)
        transitInSheepMCTSSimulation = \
        lambda state, sheepSelfAction: transit(state, [sheepSelfAction, wolfActionInSheepSimulation(state)])

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
        initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                        getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = 0.1
        maxRolloutSteps = 10
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                      rolloutHeuristic)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        getStateFromNode = lambda node: list(node.id.values())[0]

        # sample trajectory
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, sampleAction)

        # neural network init
        numStateSpace = 12
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                            'evaluateSupervisedLearning', 'NNModel')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        NNModelSaveExtension = ''
        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        startTime = time.time()

        otherAgentApproximatePolicy = lambda NNModel: stationaryAgentPolicy
        # otherAgentApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)
        SingleAgentMCTS = mcts
        prepareMultiAgentPolicy = PrepareMultiAgentNNPolicyWithAgentSelfNNGuidedMCTS(SingleAgentMCTS, otherAgentApproximatePolicy)

        # load NN
        multiAgentNNmodel = [generateModel(sharedWidths, actionLayerWidths, valueLayerWidths) for agentId in agentIds]

        restoredMultiAgentNNModel = [restoreVariables(multiAgentNNmodel[agentId], generateNNModelSavePath( {'agentId': agentId}))\
                for agentId in agentIds]

        policy = prepareMultiAgentPolicy(agentId, restoredMultiAgentNNModel)
        # sample and save trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
