import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import pathos.multiprocessing as mp
import itertools as it

from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle, DeleteUsedModel
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from exec.parallelComputing import GenerateTrajectoriesParallel


class PreprocessTrajectoriesForBuffer:
    def __init__(self, addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory):
        self.addMultiAgentValuesToTrajectory = addMultiAgentValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addMultiAgentValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        return filteredTrajectories


class TrainOneAgent:
    def __init__(self, numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets,
                 sampleBatchFromBuffer, trainNN):
        self.numTrainStepEachIteration = numTrainStepEachIteration
        self.numTrajectoriesToStartTrain = numTrajectoriesToStartTrain
        self.sampleBatchFromBuffer = sampleBatchFromBuffer
        self.processTrajectoryForPolicyValueNets = processTrajectoryForPolicyValueNets
        self.trainNN = trainNN

    def __call__(self, agentId, multiAgentNNmodel, updatedReplayBuffer):
        NNModel = multiAgentNNmodel[agentId]
        if len(updatedReplayBuffer) >= self.numTrajectoriesToStartTrain:
            for _ in range(self.numTrainStepEachIteration):
                sampledBatch = self.sampleBatchFromBuffer(updatedReplayBuffer)
                processedBatch = self.processTrajectoryForPolicyValueNets[agentId](sampledBatch)
                trainData = [list(varBatch) for varBatch in zip(*processedBatch)]
                updatedNNModel = self.trainNN(NNModel, trainData)
                NNModel = updatedNNModel

        return NNModel


def iterateTrainOneCondition(parameterOneCondition):

    numTrainStepEachIteration = int(parameterOneCondition['numTrainStepEachIteration'])
    numTrajectoriesPerIteration = int(parameterOneCondition['numTrajectoriesPerIteration'])
    dirName = os.path.dirname(__file__)

    numOfAgent = 2
    agentIds = list(range(numOfAgent))

    sheepId = 0
    wolvesId = 1

    wolfOnePosIndex = 1
    wolfTwoIndex = 2
    xPosIndex = [0, 1]
    xBoundary = [0, 600]
    yBoundary = [0, 600]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOnePosIndex, xPosIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoIndex, xPosIndex)

    maxRunningSteps = 50
    sheepAliveBonus = 1 / maxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus

    sheepTerminalPenalty = -1
    wolfTerminalReward = 1
    terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward, wolfTerminalReward]

    killzoneRadius = 80
    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
    rewardMultiAgents = [rewardSheep, rewardWolf]
    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 9
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 6
    wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))
    actionSpaceList = [sheepActionSpace, wolvesActionSpace]

    # neural network init
    numStateSpace = 6
    numSheepActionSpace = len(sheepActionSpace)
    numWolvesActionSpace = len(wolvesActionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]

    generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
    generateModelList = [generateSheepModel, generateWolvesModel]

    # replay buffer
    bufferSize = 3000
    saveToBuffer = SaveToBuffer(bufferSize)
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 256  # 256
    sampleBatchFromBuffer = SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # pre-process the trajectory for replayBuffer
    addMultiAgentValuesToTrajectory = AddValuesToTrajectory(accumulateMultiAgentRewards)
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    # pre-process the trajectory for NNTraining
    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)
    wolvesActionToOneHot = ActionToOneHot(wolvesActionSpace)
    actionToOneHotList = [sheepActionToOneHot, wolvesActionToOneHot]
    processTrajectoryForPolicyValueNets = [ProcessTrajectoryForPolicyValueNetMultiAgentReward(actionToOneHotList[agentId], agentId) for agentId in agentIds]

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

    trainReporter = TrainReporter(numTrainStepEachIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRate = 0.0001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    trainNN = Train(numTrainStepEachIteration, miniBatchSize, sampleData,
                    learningRateModifier, terminalController, coefficientController,
                    trainReporter)

    # load save dir
    numSimulations = 200
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsCenterControlObstaclePolicyNet', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsCenterControlObstaclePolicyNet', 'NNModel')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    startTime = time.time()
    trainableAgentIds = [sheepId, wolvesId]

    depth = 4
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for generateModel in generateModelList]

    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory)
    numTrajectoriesToStartTrain = 4 * miniBatchSize

    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)

    # save step 0 Model for evaluate
    for agentId in trainableAgentIds:
        NNModelPathParameters = {'iterationIndex': 0, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
        NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
        saveVariables(multiAgentNNmodel[agentId], NNModelSavePath)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForParallel = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)

    # initRreplayBuffer
    replayBuffer = []
    trajectoryBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': trajectoryBeforeTrainIndex}
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

    # delete used model for disk space
    fixedParametersForDelete = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
    toDeleteNNModelExtensionList = ['.meta', '.index', '.data-00000-of-00001']
    generatetoDeleteNNModelPathList = [GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParametersForDelete) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]

    # restore model
    restoredIteration = 0
    for agentId in trainableAgentIds:
        modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)

    modelMemorySize = 5
    modelSaveFrequency = 50
    deleteUsedModel = DeleteUsedModel(modelMemorySize, modelSaveFrequency, generatetoDeleteNNModelPathList)
    numIterations = 10000
    for iterationIndex in range(restoredIteration + 1, numIterations):

        numCpuToUseWhileTrain = int(3)
        numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
        sampleTrajectoryFileName = 'sampleMultiMCTSAgentCenterControlNetTrajCondtion.py'

        generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)
        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
        cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)

        trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
        trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        saveToPickle(trajectories, trajectorySavePath)

        preProcessedTrajectories = preprocessMultiAgentTrajectories(trajectories)
        updatedReplayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectories)

        for agentId in trainableAgentIds:

            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)

            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)
            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

            deleteUsedModel(iterationIndex, agentId)

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [1]
    manipulatedVariables['numTrajectoriesPerIteration'] = [1]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # Sample Trajectory Before Train to fill Buffer
    numTrajectoriesToStartTrain = 1024
    sampleTrajectoryFileName = 'prepareMCTSAgentCenterControlNetTraj.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8 * numCpuCores)
    print(numCpuToUse)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    iterationBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': iterationBeforeTrainIndex}
    prepareBefortrainData = True
    if prepareBefortrainData:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)

    # parallel train
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)


if __name__ == '__main__':
    main()
