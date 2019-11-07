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

from src.constrainedChasingEscapingEnv.envNoPhysics import  IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle,DeleteUsedModel
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
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

def iterateTrainOneCondition(manipulatedVariable):
    numTrainStepEachIteration = int(manipulatedVariable['numTrainStepEachIteration'])
    numTrajectoriesPerIteration = int(manipulatedVariable['numTrajectoriesPerIteration'])
    dirName = os.path.dirname(__file__)


    numOfAgent=3
    agentIds = list(range(numOfAgent))
    
    sheepId = 0
    wolfOneId = 1
    wolfTwoId = 2
    xPosIndex = [0, 1]
    xBoundary = [0,600]
    yBoundary = [0,600]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
    getWolfTwoXPos =GetAgentPosFromState(wolfTwoId, xPosIndex)


    maxRunningSteps = 150
    sheepAliveBonus = 1/maxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus

    sheepTerminalPenalty = -1
    wolfTerminalReward = 1
    terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward,wolfTerminalReward]

    killzoneRadius = 30
    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)



    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)

    rewardMultiAgents = [rewardSheep, rewardWolf,rewardWolf]
    
    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)


    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]


    # neural network init
    numStateSpace = 6
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # replay buffer
    bufferSize = 2000
    saveToBuffer = SaveToBuffer(bufferSize)
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 256#256
    sampleBatchFromBuffer = SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # pre-process the trajectory for replayBuffer
    addMultiAgentValuesToTrajectory = AddValuesToTrajectory(accumulateMultiAgentRewards)
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    # pre-process the trajectory for NNTraining
    actionToOneHot = ActionToOneHot(actionSpace)
    processTrajectoryForPolicyValueNets = [ProcessTrajectoryForPolicyValueNetMultiAgentReward(actionToOneHot, agentId) for agentId in agentIds]

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
    numSimulations = 100
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'NNModelRes')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)
    

    startTime = time.time()
    trainableAgentIds = [sheepId, wolfOneId,wolfTwoId]

    depth = 5
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in agentIds]


    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory)
    numTrajectoriesToStartTrain = 4 * miniBatchSize

    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)


    for agentId in trainableAgentIds:
        #creat step 0 for evaluate
        NNModelPathParameters = {'iterationIndex': 0, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
        NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
        saveVariables(multiAgentNNmodel[agentId], NNModelSavePath)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForParallel = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    # load trajectory function for trainBreak
    loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)

# initRreplayBuffer
    replayBuffer = []
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

# restore model
    restoredIteration = 100
    for agentId in trainableAgentIds:
        modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId,  'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)

    #delete used model for disk space
    fixedParametersForDelete = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
    toDeleteNNModelExtensionList=['.meta','.index','.data-00000-of-00001']
    generatetoDeleteNNModelPathList=[GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParametersForDelete) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]
    modelMemorySize=5
    modelSaveFrequency=50
    deleteUsedModel=DeleteUsedModel(modelMemorySize,modelSaveFrequency,generatetoDeleteNNModelPathList)

    sampleTrajectoryFileName='multiAgentTrain2SeperateResNetMultiProcessParallel.py'.py'
    print ({'iterationIndex': restoredIteration, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
    numIterations = 10000
    for iterationIndex in range(restoredIteration + 1, numIterations):

        numCpuToUseWhileTrain = int(16)
        numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
        generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)

        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}

        trajecoriesNum=0
        # prevent subprocess crack
        while trajecoriesNum!=numTrajectoriesPerIteration:
            cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)

            trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
            trajecoriesNum=len(trajectories)        
            if trajecoriesNum!=numTrajectoriesPerIteration:
                print('MISSSUBPROCESS,RETRY',trajecoriesNum)
        print('length of traj', len(trajectories))
        trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        saveToPickle(trajectories, trajectorySavePath)

        preProcessedTrajectories = preprocessMultiAgentTrajectories(trajectories)
        updatedReplayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectories)

        for agentId in trainableAgentIds:

            deleteUsedModel(iterationIndex,agentId)

            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)
            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)


            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [1]
    manipulatedVariables['numTrajectoriesPerIteration'] = [16]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    miniBatchSize = 256
    numTrajectoriesToStartTrain = 4 * miniBatchSize
    sampleTrajectoryFileName = 'prepareMultiMCTSAgent2SeperateResNetTraj.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    iterationBeforeTrainIndex=0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': iterationBeforeTrainIndex}

    prepareBefortrainData=False
    if prepareBefortrainData:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)


if __name__ == '__main__':
    # main()
    parameters={}
    parameters['numTrainStepEachIteration'] = 1
    parameters['numTrajectoriesPerIteration'] = 10
    iterateTrainOneCondition(parameters)