import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
import psutil
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import csv

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, Reset, TransitionFunction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.evaluationFunctions import GetSavePath, readParametersFromDf, LoadTrajectories
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValueFunction, \
    ApproximateActionPrior, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import GetApproximateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


def saveData(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


class GenerateTrajectories:
    def __init__(self, numTrajectoriesPerIteration, sampleTrajectory, preparePolicy, restoreModelFromDf,
                 readParametersFromDf, getTrajectorySavePath, saveData):
        self.numTrajectoriesPerIteration = numTrajectoriesPerIteration
        self.sampleTrajectory = sampleTrajectory
        self.preparePolicy = preparePolicy
        self.restoreModelFromDf = restoreModelFromDf
        self.readParametersFromDf = readParametersFromDf
        self.getTrajectorySavePath = getTrajectorySavePath
        self.saveData = saveData

    def __call__(self, oneConditionDf):
        NNModel = self.restoreModelFromDf(oneConditionDf)
        pathParameters = self.readParametersFromDf(oneConditionDf)
        parameterWithSampleIndex = lambda sampleIndex: dict(list(pathParameters.items()) + [('sampleIndex', sampleIndex)])
        allIndexPaths = [self.getTrajectorySavePath(parameterWithSampleIndex(sampleIndex)) for sampleIndex in
                         range(self.numTrajectoriesPerIteration)]
        policy = self.preparePolicy(NNModel)
        trajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrajectoriesPerIteration)]
        [self.saveData(trajectory, path) for trajectory, path in zip(trajectories, allIndexPaths)]

        return None


class ProcessTrajectoryForNN:
    def __init__(self, actionToOneHot, agentId):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class PreProcessTrajectories:
    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        processedTrajectories = [self.processTrajectoryForNN(trajectory) for trajectory in filteredTrajectories]

        return processedTrajectories


class SaveMemoryUse:
    def __init__(self, getSavePath, readParametersFromDf):
        self.getSavePath = getSavePath
        self.readParametersFromDf = readParametersFromDf

    def __call__(self, oneConditionDf):
        pathParameters = self.readParametersFromDf(oneConditionDf)
        savePath = self.getSavePath(pathParameters)
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss

        with open(savePath, 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow([memory])
        writeFile.close()


class PlotMemoryUse:
    def __init__(self, getSavePath, readParametersFromDf):
        self.getSavePath = getSavePath
        self.readParametersFromDf = readParametersFromDf

    def __call__(self, df):
        for key, grp in df.groupby('numTrajectoriesPerIteration'):
            pathParameters = self.readParametersFromDf(grp)
            savePath = self.getSavePath(pathParameters)
            print("PATH: ", savePath)
            with open(savePath, 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)
            readFile.close()
            memoryUse = [int(line[0]) for line in lines]
            plt.plot(memoryUse, marker='o', label=key)


class IterativePlayAndTrain:
    def __init__(self, numIterations, trajectorySaveDirectory, saveNNModel, getGenerateTrajectories, saveMemoryUse,
                 preProcessTrajectories, getSampleBatchFromBuffer, getTrainNN, getNNModel, loadTrajectories,
                 getSaveToBuffer):
        self.numIterations = numIterations
        self.trajectorySaveDirectory = trajectorySaveDirectory
        self.saveNNModel = saveNNModel
        self.getGenerateTrajectories = getGenerateTrajectories
        self.saveMemoryUse = saveMemoryUse
        self.preProcessTrajectories = preProcessTrajectories
        self.getSampleBatchFromBuffer = getSampleBatchFromBuffer
        self.getTrainNN = getTrainNN
        self.getNNModel = getNNModel
        self.loadTrajectories = loadTrajectories
        self.getSaveToBuffer = getSaveToBuffer

    def __call__(self, oneConditionDf):
        numTrajectoriesPerIteration = oneConditionDf.index.get_level_values('numTrajectoriesPerIteration')[0]
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]
        windowSize = oneConditionDf.index.get_level_values('windowSize')[0]

        generateTrajectories = self.getGenerateTrajectories(numTrajectoriesPerIteration)
        sampleBatchFromBuffer = self.getSampleBatchFromBuffer(miniBatchSize)
        trainNN = self.getTrainNN(learningRate)

        NNModel = self.getNNModel()
        buffer = []
        saveToBuffer = self.getSaveToBuffer(windowSize)
        startTime = time.time()
        for iterationIndex in range(self.numIterations):
            conditionDfOneIteration = pd.concat([oneConditionDf], keys=[iterationIndex], names=['iterationIndex'])
            self.saveNNModel(NNModel, conditionDfOneIteration)
            _ = generateTrajectories(conditionDfOneIteration)
            trajectories = self.loadTrajectories(conditionDfOneIteration)
            processedTrajectories = self.preProcessTrajectories(trajectories)
            updatedBuffer = saveToBuffer(buffer, processedTrajectories)
            if len(updatedBuffer) >= windowSize:
                sampledBatch = sampleBatchFromBuffer(updatedBuffer)
                trainData = [list(varBatch) for varBatch in zip(*sampledBatch)]
                updatedNNModel = trainNN(NNModel, trainData)

            NNModel = updatedNNModel
            buffer = updatedBuffer

            self.saveMemoryUse(oneConditionDf)

        endTime = time.time()
        print("Time taken for {} iterations: {} seconds".format(self.numIterations, (endTime-startTime)))


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, NNModel):
        wolfPolicy = self.getWolfPolicy(NNModel)
        sheepPolicy = self.getSheepPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrajectoriesPerIteration'] = [0, 32, 64, 128, 256]#[256]
    manipulatedVariables['miniBatchSize'] = [512]
    manipulatedVariables['learningRate'] = [0.01]
    manipulatedVariables['loadNN'] = [False]
    manipulatedVariables['windowSize'] = [1000]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getNNModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    maxRunningSteps = 25
    numSimulations = 2#50
    qPosInit = (0, 0, 0, 0)
    qPosInitNoise = 9.7
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'dump',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    NNPathFromDf = lambda oneConditionDf: getNNModelSavePath(readParametersFromDf(oneConditionDf)) 
    saveNNModel = lambda NNmodel, oneConditionDf: saveVariables(NNmodel, NNPathFromDf(oneConditionDf))


    #trajectory path to load
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'dump',
                                           'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    #load trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadData, readParametersFromDf)

    # pre-process the trajectory for training the neural network
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = -1/maxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
                                                for index in range(len(actionSpace))])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForNN(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 0
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    wolfActionInSheepMCTSSimulation = lambda state: (0, 0)
    transitInSheepMCTSSimulation = lambda state, sheepSelfAction: transit(state, [sheepSelfAction, wolfActionInSheepMCTSSimulation(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # functions to make predictions from NN
    getApproximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                                   getApproximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))


    getStateFromNode = lambda node: list(node.id.values())[0]
    getApproximateValueFromNode = lambda NNModel: \
        GetApproximateValueFromNode(getStateFromNode, ApproximateValueFunction(NNModel))

    # wrapper for MCTS
    getMCTSNNPriorValue = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                               getApproximateValueFromNode(NNModel), backup, establishPlainActionDist)

    # policy
    getWolfPolicy = lambda NNModel: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getWolfPolicy, getMCTSNNPriorValue)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    getModelPathFromDf = lambda oneConditionDf: getNNModelSavePath(readParametersFromDf(oneConditionDf))
    restoreModelFromDf = lambda oneConditionDf: restoreVariables(getNNModel(), getModelPathFromDf(oneConditionDf))
    getGenerateTrajectories = \
        lambda numTrajectoriesPerIteration: GenerateTrajectories(numTrajectoriesPerIteration, sampleTrajectory,
                                                                 preparePolicy, restoreModelFromDf, readParametersFromDf,
                                                                 getTrajectorySavePath, saveData)

    # replay buffer
    getUniformSamplingProbabilities = lambda buffer: [(1/len(buffer)) for _ in buffer]
    getSampleBatchFromBuffer = lambda miniBatchSize: SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # function to train NN model
    batchSizeForTrainFunction = 0
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
    reportInterval = 25
    numTrainStepsPerIteration = 1
    trainReporter = TrainReporter(numTrainStepsPerIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda learningRate: Train(numTrainStepsPerIteration, batchSizeForTrainFunction, sampleData,
                                                      learningRateModifier(learningRate),
                                                      terminalController, coefficientController,
                                                      trainReporter)

    # recording memory usage
    memoryRecordDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'dump')
    memoryRecordFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    memoryRecordExtension = '.csv'
    getMemoryRecordSavePath = GetSavePath(memoryRecordDirectory, memoryRecordExtension, memoryRecordFixedParameters)

    saveMemoryUse = SaveMemoryUse(getMemoryRecordSavePath, readParametersFromDf)

    # functions to iteratively play and train the NN
    numIterations = 30
    iterativePlayAndTrain = IterativePlayAndTrain(numIterations, trajectorySaveDirectory, saveNNModel,
                                                  getGenerateTrajectories, saveMemoryUse, preProcessTrajectories,
                                                  getSampleBatchFromBuffer, getTrainNN, getNNModel, loadTrajectories,
                                                  SaveToBuffer)

    performanceDf = toSplitFrame.groupby(levelNames).apply(iterativePlayAndTrain)

    # plot
    plotMemoryUse = PlotMemoryUse(getMemoryRecordSavePath, readParametersFromDf)
    plt.xlabel("Iteration")
    plt.ylabel("memory usage")
    plt.title("Memory usage vs. iteration")
    plotMemoryUse(toSplitFrame)
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()
