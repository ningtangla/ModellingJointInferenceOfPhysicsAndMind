import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from subprocess import Popen, PIPE
import json
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
from psutil import virtual_memory, disk_usage

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.evaluationFunctions import GetSavePath, readParametersFromDf, LoadTrajectories
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


class GenerateTrajectoriesParallel:
    def __init__(self, codeFileName, numSample, readParametersFromDf):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.readParametersFromDf = readParametersFromDf

    def __call__(self, oneConditionDf):
        sampleIdStrings = list(map(str, range(self.numSample)))
        parameters = self.readParametersFromDf(oneConditionDf)
        parametersString = json.dumps(parameters)
        cmdList = [['python3', self.codeFileName, parametersString, sampleIndex] for sampleIndex in sampleIdStrings]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.wait()
        return cmdList

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


class IterativePlayAndTrain:
    def __init__(self, windowSize, numIterations, trajectoriesPath, initializedNNModel, saveNNModel,
                 getGenerateTrajectoriesParallel, loadTrajectories, preProcessTrajectories, saveToBuffer,
                 getSampleBatchFromBuffer, getTrainNN):
        self.windowSize = windowSize
        self.numIterations = numIterations
        self.trajectoriesPath = trajectoriesPath
        self.initializedNNModel = initializedNNModel
        self.saveNNModel = saveNNModel
        self.getGenerateTrajectoriesParallel = getGenerateTrajectoriesParallel
        self.loadTrajectories = loadTrajectories
        self.preProcessTrajectories = preProcessTrajectories
        self.saveToBuffer = saveToBuffer
        self.getSampleBatchFromBuffer = getSampleBatchFromBuffer
        self.getTrainNN = getTrainNN

    def __call__(self, oneConditionDf):
        numTrajectoriesPerIteration = oneConditionDf.index.get_level_values('numTrajectoriesPerIteration')[0]
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]

        generateTrajectoriesParallel = self.getGenerateTrajectoriesParallel(numTrajectoriesPerIteration)
        sampleBatchFromBuffer = self.getSampleBatchFromBuffer(miniBatchSize)
        trainNN = self.getTrainNN(learningRate)

        NNModel = self.initializedNNModel
        buffer = []
        usedVirtualMemory = []
        # percentDiskUsage = []
        for iterationIndex in range(self.numIterations):
            conditionDfOneIteration = pd.concat([oneConditionDf], keys = [iterationIndex], names = ['iterationIndex'])
            self.saveNNModel(NNModel, conditionDfOneIteration)
            print("iteration: ", iterationIndex)
            cmdListGenerateTra = generateTrajectoriesParallel(conditionDfOneIteration)
            trajectories = self.loadTrajectories(conditionDfOneIteration)
            processedTrajectories = self.preProcessTrajectories(trajectories)
            updatedBuffer = self.saveToBuffer(buffer, processedTrajectories)
            # if len(updatedBuffer) >= self.windowSize:
            sampledBatch = sampleBatchFromBuffer(updatedBuffer) #
            trainData = [list(varBatch) for varBatch in zip(*sampledBatch)] #
            updatedNNModel = trainNN(NNModel, trainData)    #
            NNModel = updatedNNModel    #

            buffer = updatedBuffer

            virtualMemory = virtual_memory()
            usedVirtualMemory.append(virtualMemory.used)
            # diskUsage = disk_usage(self.trajectoriesPath)
            # percentDiskUsage.append(diskUsage.percent)

        # plt.plot(percentDiskUsage, marker='o')
        # plt.title('percent disk usage')
        # plt.xlabel('iteration')
        # plt.show()
        plt.plot(usedVirtualMemory, marker='o')
        plt.title('virtual memory')
        plt.xlabel('iteration')
        plt.show()


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrajectoriesPerIteration'] = [25]#[256]
    manipulatedVariables['miniBatchSize'] = [512]
    manipulatedVariables['learningRate'] = [0.01]

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
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    maxRunningSteps = 10
    numSimulations = 2#50
    qPosInit = (0, 0, 0, 0)
    qPosInitNoise = 9.7
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    NNPathFromDf = lambda oneConditionDf: getNNModelSavePath(readParametersFromDf(oneConditionDf)) 
    saveNNModel = lambda NNmodel, oneConditionDf: saveVariables(NNmodel, NNPathFromDf(oneConditionDf)) 
  
    #generate trajectory 
    generateTrajectoriesCodeName = 'generateTrajectoryMCTSNNPriorRolloutPolicySheepChaseWolfMujoco.py'
    getGenerateTrajectoriesParallel = lambda numTrajectoriesPerIteration: GenerateTrajectoriesParallel(generateTrajectoriesCodeName, numTrajectoriesPerIteration,
            readParametersFromDf)
    
    #trajectory path to load
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
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

    # replay buffer
    windowSize = 5000
    saveToBuffer = SaveToBuffer(windowSize)
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
    # functions to iteratively play and train the NN
    numIterations = 1#100#40#150
    iterativePlayAndTrain = IterativePlayAndTrain(windowSize, numIterations, trajectorySaveDirectory, initializedNNModel,
                                                  saveNNModel, getGenerateTrajectoriesParallel, loadTrajectories,
                                                  preProcessTrajectories, saveToBuffer, getSampleBatchFromBuffer, getTrainNN)
    startTime = time.time()
    performanceDf = toSplitFrame.groupby(levelNames).apply(iterativePlayAndTrain)
    endTime = time.time()
    print("time for {} iterations = {}".format(numIterations, (endTime-startTime)))
    # plt.plot(performanceDf.values[0])
    # plt.show()


if __name__ == '__main__':
    main()
