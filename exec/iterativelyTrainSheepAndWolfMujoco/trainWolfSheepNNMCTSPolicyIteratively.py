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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValueFunction, \
    ApproximateActionPrior, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories, ActionToOneHot
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


def main():
    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    
    #MDP function
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 1
    qPosInitNoise = 9.7
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 1 / maxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    
    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, playIsTerminal, numSimulationFrames)

    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # NNGuidedMCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    terminalReward = 1
    terminalPenalty = -1

    transitInSheepMCTS = lambda state, selfAction: transit(state, [selfAction, wolfNNPolicy(state)])#self other
    approximateActionPrior = ApproximatePolicy(actionSpace)
    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInMCTS, approximateActionPrior)

    getExpandNNPrior = lambda NNModel: Expand(isTerminal, InitializeChildren)
    getStateFromNode = lambda node: list(node.id.values())[0]
    getEstimateValue = lambda NNModel: EstimateValueFromNode(terminalReward, isTerminal, getStateFromNode, ApproximateValueFunction(NNModel))

    numSimulations = 200  
    getMCTSNNPriorValue = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                               getEstimateValue(NNModel), backup, establishPlainActionDist)

    # sample trajectory
    maxRunningSteps = 25
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # replay buffer
    bufferSize = 2000
    saveToBuffer = SaveToBuffer(bufferSize)
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 64
    sampleBatchFromBuffer = SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # pre-process the trajectory for training the neural network
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    wolfActionToOneHot = ActionToOneHot(wolfActionSpace)
    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)

    processSheepTrajectoryForNN = ProcessTrajectoryForNN(sheepActionToOneHot, sheepId)
    preProcessSheepTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                         processSheepTrajectoryForNN)

    processWolfTrajectoryForNN = ProcessTrajectoryForNN(wolfActionToOneHot, wolfId)
    preProcessWolfTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                        processWolfTrajectoryForNN)

    # function to train NN model
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 1
    numTrainStepsPerIteration = 1
    trainReporter = TrainReporter(numTrainStepsPerIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRate = 0.001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = Train(numTrainStepsPerIteration, miniBatchSize, sampleData,
                                            learningRateModifier(learningRate),
                                            terminalController, coefficientController,
                                            trainReporter)

    trainNN = getTrainNN(learningRate)
    # functions to iteratively play and train the NN
    combineDict = lambda dict1, dict2: dict(
        list(dict1.items()) + list(dict2.items()))

    generatePathParametersAtIteration = lambda iterationIndex: \
        combineDict(NNFixedParameters,
                    {'iteration': iterationIndex})


# load save dir
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
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
    # wolfBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
    #                                                 'SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')
    # baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
    #                           'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
    #                           'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}
    # getWolfBaselineModelSavePath = GetSavePath(wolfBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)
    # baselineModelTrainSteps = 1000
    # wolfBaselineNNModelSavePath = getWolfBaselineModelSavePath({'trainSteps': baselineModelTrainSteps})
    # wolfBaselienModel = restoreVariables(initializedNNModel, wolfBaselineNNModelSavePath)
    # wolfNNModel = wolfBaselienModel
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

# load sheep baseline for init iteration
    # sheepBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
    #                                                 'SheepWolfBaselinePolicy', 'sheepBaselineNNPolicy')
    # baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
    #                           'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
    #                           'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}
    # getSheepBaselineModelSavePath = GetSavePath(sheepBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)
    # baselineModelTrainSteps = 1000
    # sheepBaselineNNModelSavePath = getSheepBaselineModelSavePath({'trainSteps': baselineModelTrainSteps})
    # sheepBaselienModel = restoreVariables(initializedNNModel, sheepBaselineNNModelSavePath)
    # sheepNNModel = sheepBaselienModel
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    
    sheepBuffer = []
    wolfBuffer = []
    
    numTrajectoriesPerIteration = 1
    learningThresholdFactor = 4
    numIterations = 2000  
    startTime = time.time()
    for iterationIndex in range(numIterations):
        print("ITERATION INDEX: ", iterationIndex)
        pathParametersAtIteration = generatePathParametersAtIteration(iterationIndex)

# sheep play
        # approximateWolfPolicy = ApproximatePolicy(wolfNNModel, wolfActionSpace)

        # getSheepPolicy = getSheepMCTS(approximateWolfPolicy)
        # sheepPolicy = getSheepPolicy(sheepNNModel)
        # wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}
        # policyForSheepTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        # trajectoriesForSheepTrain = [sampleTrajectory(policyForSheepTrain) for _ in range(numTrajectoriesPerIteration)]
        # getSheepTrajectorySavePath = GetSavePath(
        #     trajectoriesForSheepTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)

        # sheepDataSetPath = getSheepTrajectorySavePath(NNFixedParameters)
        # saveToPickle(trajectoriesForSheepTrain, sheepDataSetPath)

        # # sheepDataSetTrajectories = loadFromPickle(sheepDataSetPath)
        # sheepDataSetTrajectories = trajectoriesForSheepTrain

        # processedSheepTrajectories = preProcessSheepTrajectories(sheepDataSetTrajectories)
        # updatedSheepBuffer = saveToBuffer(sheepBuffer, processedSheepTrajectories)
        # if len(updatedSheepBuffer) >= learningThresholdFactor * miniBatchSize:
        #     sheepSampledBatch = sampleBatchFromBuffer(updatedBuffer)
        #     sheepTrainData = [list(varBatch) for varBatch in zip(*sheepSampledBatch)]
        #     updatedSheepNNModel = trainNN(sheepNNModel, sheepTrainData)
        #     sheepNNModel = updatedSheepNNModel

        # sheepBuffer = updatedSheepBuffer

# sheep train

        # getSheepModelSavePath = GetSavePath(
        #     sheepNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        # sheepNNModelSavePaths = getSheepModelSavePath({'killzoneRadius': killzoneRadius})
        # savedVariablesSheep = saveVariables(sheepNNModel, sheepNNModelSavePaths)


# wolf play
        approximateSheepPolicy = lambda state: (0, 0)

        # approximateSheepPolicy = ApproximatePolicy(sheepNNModel, sheepActionSpace)
        getWolfPolicy = getMCTSNNPriorValue(approximateSheepPolicy)
        wolfPolicy = getWolfPolicy(wolfNNModel)
        sheepPolicy = lambda state: {approximateSheepPolicy(state): 1}
        policyForWolfTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        trajectoriesForWolfTrain = [sampleTrajectory(policyForWolfTrain) for _ in range(numTrajectoriesPerIteration)]

        getWolfTrajectorySavePath = GetSavePath(
            trajectoriesForWolfTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)
        wolfDataSetPath = getWolfTrajectorySavePath(NNFixedParameters)
        saveToPickle(trajectoriesForWolfTrain, wolfDataSetPath)

        wolfDataSetTrajectories = loadFromPickle(wolfDataSetPath)

        processedWolfTrajectories = preProcessWolfTrajectories(wolfDataSetTrajectories)
        updatedWolfBuffer = saveToBuffer(wolfBuffer, processedWolfTrajectories)
        if len(updatedWolfBuffer) >= learningThresholdFactor * miniBatchSize:
            wolfSampledBatch = sampleBatchFromBuffer(updatedBuffer)
            wolfTrainData = [list(varBatch) for varBatch in zip(*wolfSampledBatch)]
            updatedWolfNNModel = trainNN(wolfNNModel, wolfTrainData)
            wolfNNModel = updatedWolfNNModel

        wolfBuffer = updatedWolfBuffer
# wolf train

        getWolfModelSavePath = GetSavePath(
            wolfNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        wolfNNModelSavePaths = getWolfModelSavePath({'killzoneRadius': killzoneRadius})
        savedVariablesWolf = saveVariables(wolfNNModel, wolfNNModelSavePaths)

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        self.numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()
