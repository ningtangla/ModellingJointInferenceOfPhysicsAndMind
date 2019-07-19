import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco
import random

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyNet import GenerateModel, Train, saveVariables, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist, RollOut
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


class ProcessTrajectoryForNN:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [
            pair for trajectory in trajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(
            lambda pair: pair[self.actionIndex] is not None and pair[0][1][2] < 9.7, stateActionPairs))
        print("{} data points remain after filtering".format(
            len(stateActionPairsFiltered)))
        stateActionPairsProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]))
                                     for state, actions, actionDist in stateActionPairsFiltered]

        random.shuffle(stateActionPairsProcessed)
        trainData = [[state for state, action in stateActionPairsProcessed],
                     [action for state, action in stateActionPairsProcessed]]

        return trainData


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(
            self.actionSpace[index])).all() else 0 for index in range(len(self.actionSpace))])
        return oneHotAction


class ApproximatePolicy:
    def __init__(self, model, actionSpace):
        self.actionSpace = actionSpace
        self.model = model

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
        graph = self.model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionIndices_ = graph.get_collection_ref("actionIndices")[0]
        actionIndices = self.model.run(
            actionIndices_, feed_dict={state_: stateBatch})
        actionBatch = [self.actionSpace[i] for i in actionIndices]
        if len(actionBatch) == 1:
            actionBatch = actionBatch[0]
        return actionBatch


class GetMcts():
    def __init__(self, actionSpace, numSimulations, selectChild, isTerminal, maxRolloutSteps, rewardFunction, rolloutHeuristic, transit):
        self.numSimulations = numSimulations
        self.selectChild = selectChild
        self.actionSpace = actionSpace
        self.isTerminal = isTerminal
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunction = rewardFunction
        self.rolloutHeuristic = rolloutHeuristic
        self.transit = transit
        self.numActionSpace = len(actionSpace)

    def __call__(self, NNPolicy):
        getUniformActionPrior = lambda state: {action: 1 / self.numActionSpace for action in self.actionSpace}
        transitInMCTS = lambda state, action: self.transit(state, [action, NNPolicy(state)])
        initializeChildrenUniformPrior = InitializeChildren(self.actionSpace, transitInMCTS,
                                                            getUniformActionPrior)

        expand = Expand(self.isTerminal, initializeChildrenUniformPrior)
        rolloutPolicy = lambda state: self.actionSpace[np.random.choice(range(self.numActionSpace))]
        rollout = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, self.rewardFunction, self.isTerminal,
                          self.rolloutHeuristic)
        mcts = MCTS(self.numSimulations, self.selectChild, expand, rollout, backup, establishPlainActionDist)
        return mcts


def main():
    numIterations = 10
    numTrajectoriesPerIteration = 2
    numSimulations = 10
    maxRunningSteps = 10
    trainSteps = 1000

    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = 0.1

    NNFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                         'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                         'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    NNModelSaveExtension = ''

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    qPosInit = (0, 0, 0, 0)
    qVelInit = (0, 0, 0, 0)
    numAgents = 2
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    actionIndex = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    numActionSpace = len(sheepActionSpace)

    # neural network init and save path
    numStateSpace = 12
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    initializedNNModel = generatePolicyNet(hiddenWidths)

    # train NN models
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10

    trainNN = Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval, summaryOn=False, testData=None)

    wolfActionToOneHot = ActionToOneHot(wolfActionSpace)
    preProcessWolfTrajectories = ProcessTrajectoryForNN(wolfId, actionIndex, wolfActionToOneHot)

    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)
    preProcessSheepTrajectories = ProcessTrajectoryForNN(sheepId, actionIndex, sheepActionToOneHot)

    # MCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    aliveBonus = 0.05
    deathPenalty = -1
    sheepRewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)
    wolfRewardFunction = RewardFunctionCompete(-aliveBonus, -deathPenalty, isTerminal)

    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)

    getSheepMCTS = GetMcts(sheepActionSpace, numSimulations, selectChild, isTerminal, maxRolloutSteps, sheepRewardFunction, rolloutHeuristic, transit)

    getWolfMCTS = GetMcts(wolfActionSpace, numSimulations, selectChild, isTerminal, maxRolloutSteps, wolfRewardFunction, rolloutHeuristic, transit)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # functions to iteratively play and train the NN
    combineDict = lambda dict1, dict2: dict(
        list(dict1.items()) + list(dict2.items()))

    generatePathParametersAtIteration = lambda iterationIndex: \
        combineDict(NNFixedParameters,
                    {'iteration': iterationIndex})

# load save dir
    trajectorySaveExtension = '.pickle'

    trajectoriesForSheepTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                          'trajectoriesForSheepTrain')
    if not os.path.exists(trajectoriesForSheepTrainSaveDirectory):
        os.makedirs(trajectoriesForSheepTrainSaveDirectory)

    sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'SheepWolfIterationPolicy', 'sheepPolicy')
    if not os.path.exists(sheepNNModelSaveDirectory):
        os.makedirs(sheepNNModelSaveDirectory)

    trajectoriesForWolfTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                         'trajectoriesForWolfTrain')
    if not os.path.exists(trajectoriesForWolfTrainSaveDirectory):
        os.makedirs(trajectoriesForWolfTrainSaveDirectory)

    wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                            'WolfWolfIterationPolicy', 'wolfPolicy')
    if not os.path.exists(wolfNNModelSaveDirectory):
        os.makedirs(wolfNNModelSaveDirectory)


# load wolf baseline for init iteration
    wolfBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                    'SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')
    baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
                              'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
                              'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}
    getWolfBaselineModelSavePath = GetSavePath(wolfBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)
    baselineModelTrainSteps = 1000
    wolfBaselineNNModelSavePath = getWolfBaselineModelSavePath({'trainSteps': baselineModelTrainSteps})
    wolfBaselienModel = restoreVariables(initializedNNModel, wolfBaselineNNModelSavePath)

    wolfNNModel = wolfBaselienModel
    startTime = time.time()
    for iterationIndex in range(numIterations):
        print("ITERATION INDEX: ", iterationIndex)
        pathParametersAtIteration = generatePathParametersAtIteration(iterationIndex)

# sheep play
        approximateWolfPolicy = ApproximatePolicy(wolfNNModel, wolfActionSpace)
        mctsSheep = getSheepMCTS(approximateWolfPolicy)
        wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}
        policyForSheepTrain = lambda state: [mctsSheep(state), wolfPolicy(state)]

        trajectoriesForSheepTrain = [sampleTrajectory(policyForSheepTrain) for _ in range(numTrajectoriesPerIteration)]
        getSheepTrajectorySavePath = GetSavePath(
            trajectoriesForSheepTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)

        sheepDataSetPath = getSheepTrajectorySavePath(NNFixedParameters)
        saveToPickle(trajectoriesForSheepTrain, sheepDataSetPath)

        # sheepDataSetTrajectories = loadFromPickle(sheepDataSetPath)
        sheepDataSetTrajectories = trajectoriesForSheepTrain
        sheepTrainData = preProcessSheepTrajectories(sheepDataSetTrajectories)

# sheep train
        updatedSheepNNModel = trainNN(generatePolicyNet(hiddenWidths), sheepTrainData)

        getSheepModelSavePath = GetSavePath(
            sheepNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        sheepNNModelSavePaths = getSheepModelSavePath({'trainSteps': trainSteps})
        savedVariablesSheep = saveVariables(updatedSheepNNModel, sheepNNModelSavePaths)


# wolf play
        sheepNNModel = updatedSheepNNModel

        approximateSheepPolicy = ApproximatePolicy(sheepNNModel, sheepActionSpace)
        mctsWolf = getWolfMCTS(approximateSheepPolicy)
        sheepPolicy = lambda state: {approximateSheepPolicy(state): 1}
        policyForWolfTrain = lambda state: [sheepPolicy(state), mctsWolf(state)]

        trajectoriesForWolfTrain = [sampleTrajectory(policyForWolfTrain) for _ in range(numTrajectoriesPerIteration)]

        getWolfTrajectorySavePath = GetSavePath(
            trajectoriesForWolfTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)
        wolfDataSetPath = getWolfTrajectorySavePath(NNFixedParameters)
        saveToPickle(trajectoriesForWolfTrain, wolfDataSetPath)

        # wolfDataSetTrajectories = loadFromPickle(wolfDataSetPath)
        wolfDataSetTrajectories = trajectoriesForWolfTrain
        wolfTrainData = preProcessWolfTrajectories(wolfDataSetTrajectories)

# wolf train
        updatedWolfNNModel = trainNN(generatePolicyNet(hiddenWidths), wolfTrainData)

        getWolfModelSavePath = GetSavePath(
            wolfNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        wolfNNModelSavePaths = getWolfModelSavePath({'trainSteps': trainSteps})
        savedVariablesWolf = saveVariables(updatedWolfNNModel, wolfNNModelSavePaths)

        wolfNNModel = updatedWolfNNModel

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        self.numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()
