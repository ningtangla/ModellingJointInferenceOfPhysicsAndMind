import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

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


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, NNModel):
        wolfPolicy = self.getWolfPolicy(NNModel)
        sheepPolicy = self.getSheepPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, numTrajectoriesPerIteration, sampleTrajectory, preparePolicy, saveAllTrajectories):
        self.numTrajectoriesPerIteration = numTrajectoriesPerIteration
        self.sampleTrajectory = sampleTrajectory
        self.preparePolicy = preparePolicy
        self.saveAllTrajectories = saveAllTrajectories

    def __call__(self, iteration, NNModel, pathParameters):
        policy = self.preparePolicy(NNModel)
        trajectories = [self.sampleTrajectory(
            policy) for trial in range(self.numTrajectoriesPerIteration)]
        self.saveAllTrajectories(trajectories, pathParameters)

        return None


class ProcessTrajectoryForNN:
    def __init__(self, actionToOneHot, agentId):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist: \
            (np.asarray(state).flatten(), self.actionToOneHot(
                actions[self.agentId]))
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class IterativePlayAndTrain:
    def __init__(self, numIterations, learningThresholdFactor, saveNNModel, getGenerateTrajectories,
                 preProcessTrajectories, getSampleBatchFromBuffer, getTrainNN, getNNModel, loadTrajectories,
                 getSaveToBuffer, generatePathParametersAtIteration, getModelSavePath, restoreVariables):
        self.numIterations = numIterations
        self.learningThresholdFactor = learningThresholdFactor
        self.saveNNModel = saveNNModel
        self.getGenerateTrajectories = getGenerateTrajectories
        self.preProcessTrajectories = preProcessTrajectories
        self.getSampleBatchFromBuffer = getSampleBatchFromBuffer
        self.getTrainNN = getTrainNN
        self.getNNModel = getNNModel
        self.loadTrajectories = loadTrajectories
        self.getSaveToBuffer = getSaveToBuffer
        self.generatePathParametersAtIteration = generatePathParametersAtIteration
        self.getModelSavePath = getModelSavePath
        self.restoreVariables = restoreVariables

    def __call__(self, oneConditionDf):
        numTrajectoriesPerIteration = oneConditionDf.index.get_level_values(
            'numTrajectoriesPerIteration')[0]
        generateTrajectories = self.getGenerateTrajectories(
            numTrajectoriesPerIteration)
        trainNN = self.getTrainNN(learningRate)
        NNModel = self.getNNModel()
        startTime = time.time()

        for iterationIndex in range(self.numIterations):
            print("ITERATION INDEX: ", iterationIndex)
            pathParametersAtIteration = self.generatePathParametersAtIteration(
                oneConditionDf, iterationIndex)
            modelSavePath = self.getModelSavePath(pathParametersAtIteration)

            self.saveNNModel(NNModel, pathParametersAtIteration)
            generateTrajectories(iterationIndex, NNModel,
                                 pathParametersAtIteration)
            trajectories = self.loadTrajectories(pathParametersAtIteration)
            processedTrajectories = self.preProcessTrajectories(trajectories)

            trainData = ProcessTrajectoryForNN(processedTrajectories)
            updatedNNModel = trainNN(NNModel, trainData)
            NNModel = updatedNNModel

        endTime = time.time()
        print("Time taken for {} iterations: {} seconds".format(
            self.numIterations, (endTime - startTime)))


def main():
    numIterations = 1000
    numTrajectoriesPerIteration = 100
    numSimulations = 150
    maxRunningSteps = 30

    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = 0.1

    NNFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                         'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                         'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    NNModelSaveExtension = ''

    dirName = os.path.dirname(__file__)
    sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'trainNNEscapePolicyMujoco', 'trainedNNModels')
    wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                            'trainNNChasePolicyMujoco', 'trainedNNModels')

    getSheepModelSavePath = GetSavePath(sheepNNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # Mujoco environment
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    qPosInit = (0, 0, 0, 0)
    qVelInit = (0, 0, 0, 0)
    numAgents = 2
    qVelInitNoise = 8
    qPosInitNoise = 9.7
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    numActionSpace = len(sheepActionSpace)

    # neural network init and save path
    numStateSpace = 12
    numActionSpace = len(sheepActionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    initializedNNModel = generatePolicyNet(hiddenWidths)

    # train NN models
    trainSteps = 1000

    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    trainNN = Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval, summaryOn=False, testData=None)

    # MCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1 / numActionSpace for action in sheepActionSpace}

    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)

    sheepRolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
    wolfRolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # functions to iteratively play and train the NN
    combineDict = lambda dict1, dict2: dict(
        list(dict1.items()) + list(dict2.items()))

    generatePathParametersAtIteration = lambda iterationIndex: \
        combineDict(NNFixedParameters,
                    {'iteration': iterationIndex})

# load wolf baseline for init iteration
    wolfBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                    'SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')

    baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
                              'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
                              'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}

    getWolfBaselineModelSavePath = GetSavePath(wolfBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)

    wolfBaselineNNModelSavePath = getWolfBaselineModelSavePath({'trainSteps': trainSteps})
    wolfBaselienModel = restoreVariables(initializedNNModel, wolfBaselineNNModelSavePath)
    approximateWolfBaselinePolicy = ApproximatePolicy(wolfBaselienModel, wolfActionSpace)
    wolfBaselineNNPolicy = lambda state: {approximateWolfBaselinePolicy(state): 1}

    approximateWolfPolicy = approximateWolfBaselinePolicy

    for iterationIndex in range(numIterations):
        print("ITERATION INDEX: ", iterationIndex)
        pathParametersAtIteration = generatePathParametersAtIteration(iterationIndex)


# sheep play
        transitInSheepMCTS = lambda state, action: transit(state, [action, approximateWolfPolicy(state)])
        initializeChildrenUniformPriorForSheep = InitializeChildren(sheepActionSpace, transitInSheepMCTS,
                                                                    getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPriorForSheep)
        sheepRollout = RollOut(sheepRolloutPolicy, maxRolloutSteps, transitInSheepMCTS, rewardFunction, isTerminal,
                               rolloutHeuristic)
        mctsSheep = MCTS(numSimulations, selectChild, expand, sheepRollout, backup, establishPlainActionDist)

        wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}
        policyForSheepTrain = lambda state: [mctsSheep(state), wolfPolicy(state)]

        trajectoriesForSheepTrain = [sampleTrajectory(policyForSheepTrain) for _ in range(numTrajectoriesPerIteration)]
        trajectoriesForSheepTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                              'trajectoriesForSheepTrain')
        if not os.path.exists(trajectoriesForSheepTrainSaveDirectory):
            os.makedirs(trajectoriesForSheepTrainSaveDirectory)
        trajectorySaveExtension = '.pickle'
        getSheepTrajectorySavePath = GetSavePath(
            trajectoriesForSheepTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)
        sheepDataSetPath = getSheepTrajectorySavePath(NNFixedParameters)
        sheepDataSetTrajectories = loadFromPickle(sheepDataSetPath)

        actionIndex = 1
        sheepActionToOneHot = ActionToOneHot(sheepActionSpace)
        preProcessSheepTrajectories = ProcessTrajectoryForNN(sheepActionToOneHot, agentId)
        sheepTrainData = preProcessSheepTrajectories(sheepDataSetTrajectories)

        # preProcessSheepTrajectories = PreProcessTrajectories(
        #     sheepId, actionIndex, sheepActionToOneHot)
        # sheepStateActionPairsProcessed = preProcessSheepTrajectories(
        #     sheepDataSetTrajectories)
        # random.shuffle(sheepStateActionPairsProcessed)
        # sheepTrainData = [[state for state, action in sheepStateActionPairsProcessed],
        #                   [action for state, action in sheepStateActionPairsProcessed]]


# sheep train
        updatedSheepNNModel = trainNN(generatePolicyNet(hiddenWidths), sheepTrainData)
        # NNmodel save path
        sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                 'SheepWolfIterationPolicy', 'sheepPolicy')
        if not os.path.exists(sheepNNModelSaveDirectory):
            os.makedirs(sheepNNModelSaveDirectory)

        getSheepModelSavePath = GetSavePath(
            sheepNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)

        sheepNNModelSavePaths = getSheepModelSavePath({'trainSteps': trainSteps})

        # save trained model variables
        savedVariablesSheep = saveVariables(updatedSheepNNModel, sheepNNModelSavePaths)

        SheepNNModel = updatedSheepNNModel

###############
# wolf play

        approximateSheepPolicy = ApproximatePolicy(SheepNNModel, sheepActionSpace)
        transitInWolfMCTS = lambda state, action: transit(state, [action, approximateSheepPolicy(state)])
        initializeChildrenUniformPriorForWolf = InitializeChildren(sheepActionSpace, transitInWolfMCTS,
                                                                   getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPriorForWolf)

        wolfRollout = RollOut(wolfRolloutPolicy, maxRolloutSteps, transitInWolfMCTS, rewardFunction, isTerminal,
                              rolloutHeuristic)
        mctsWolf = MCTS(numSimulations, selectChild, expand, wolfRollout, backup, establishPlainActionDist)

        policyForWolfTrain = lambda state: [sheepPolicy(state), mctsWolf(state)]

        trajectoriesForWolfTrain = [sampleTrajectory(policyForWolfTrain) for _ in range(numTrajectoriesPerIteration)]

        trajectoriesForWolfTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                             'trajectoriesForWolfTrain')
        if not os.path.exists(trajectoriesForWolfTrainSaveDirectory):
            os.makedirs(trajectoriesForWolfTrainSaveDirectory)

        trajectorySaveExtension = '.pickle'
        getWolfTrajectorySavePath = GetSavePath(
            trajectoriesForWolfTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)
        wolfDataSetPath = getWolfTrajectorySavePath(NNFixedParameters)
        wolfDataSetTrajectories = loadFromPickle(wolfDataSetPath)

        actionIndex = 1
        wolfActionToOneHot = ActionToOneHot(wolfActionSpace)
        preProcessWolfTrajectories = ProcessTrajectoryForNN(wolfActionToOneHot, agentId)
        wolfTrainData = preProcessWolfTrajectories(wolfDataSetTrajectories)

        # preProcessWolfTrajectories = PreProcessTrajectoriesNN(
        #     wolfId, actionIndex, wolfActionToOneHot)
        # wolfStateActionPairsProcessed = preProcessWolfTrajectories(
        #     wolfDataSetTrajectories)
        # random.shuffle(wolfStateActionPairsProcessed)
        # wolfTrainData = [[state for state, action in wolfStateActionPairsProcessed],
        #                  [action for state, action in wolfStateActionPairsProcessed]]


# wolf train
        updatedWolfNNModel = trainNN(generatePolicyNet(hiddenWidths), sheepTrainData)
        # NNmodel save path
        wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                'WolfWolfIterationPolicy', 'wolfPolicy')
        if not os.path.exists(wolfNNModelSaveDirectory):
            os.makedirs(wolfNNModelSaveDirectory)

        getWolfModelSavePath = GetSavePath(
            wolfNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)

        wolfNNModelSavePaths = getWolfModelSavePath({'trainSteps': trainSteps})

        # save trained model variables
        savedVariablesWolf = saveVariables(updatedWolfNNModel, wolfNNModelSavePaths)

        wolfNNModel = updatedWolfNNModel
        approximateWolfPolicy = ApproximatePolicy(updatedWolfNNModel, wolfActionSpace)


if __name__ == '__main__':
    main()
