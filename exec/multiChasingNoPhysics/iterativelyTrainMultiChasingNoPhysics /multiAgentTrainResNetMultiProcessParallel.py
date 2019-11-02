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
# from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysics, Reset,IsTerminal,StayInBoundaryByReflectVelocity

from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
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
def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [1]
    manipulatedVariables['numTrajectoriesPerIteration'] = [1,10]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.8*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    #trainedModels = [trainPool.apply_async(trainModelForConditions, (parameters,)) for parameters in parametersAllCondtion]
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)


def iterateTrainOneCondition(parameters):
    # Mujoco environment
    numTrainStepEachIteration = int(parameters['numTrainStepEachIteration'])
    numTrajectoriesPerIteration = int(parameters['numTrajectoriesPerIteration'])
    dirName = os.path.dirname(__file__)

    # MDP function
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
    # preyPowerRatio = 3
    # sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    # predatorPowerRatio = 2
    # wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    


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
    
    #frequencyVersion: delete used model for disk space
    toDeleteNNModelExtensionList=['.meta','.index','.data-00000-of-00001']
    generatetoDeleteNNModelPathList=[GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParameters) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]


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

    restoredIteration = 100
    numIterations = 10000
    modelSaveFrequency=250
    modelMemorySize=10
    #serach restroeIteration
    # while restoredIteration<numIterations:
        
    #     toRestoreTrajectoryPathParameters = {'iterationIndex': restoredIteration+modelSaveFrequency, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
    #     restoreTrajectoryPath = generateTrajectorySavePath(toRestoreTrajectoryPathParameters)
        
    #     if  not os.path.isfile(restoreTrajectoryPath):
    #         restoredIteration=restoredIteration+modelSaveFrequency
    #         while restoredIteration<numIterations:
    #             restoredIteration=restoredIteration-1
    #             toRestoreTrajectoryPathParameters = {'iterationIndex': restoredIteration, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
    #             restoreTrajectoryPath = generateTrajectorySavePath(toRestoreTrajectoryPathParameters)
    #             if  os.path.isfile(restoreTrajectoryPath):
    #                 break
    #         break
    #     restoredIteration=restoredIteration+modelSaveFrequency


    for agentId in trainableAgentIds:
        modelPathBeforeTrain = generateNNModelSavePath({'iterationIndex': 0, 'agentId': agentId})
        saveVariables(multiAgentNNmodel[agentId], modelPathBeforeTrain)
        #creat step 0 for evaluate
        NNModelPathParameters = {'iterationIndex': 0, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
        NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
        saveVariables(multiAgentNNmodel[agentId], NNModelSavePath)

    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMultiMCTSAgentResNetTrajCondtion.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    trajectoryBeforeTrainPathParamters = {'iterationIndex': 0}
    if restoredIteration == 0:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)


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
    print(restoredIteration)
    for agentId in trainableAgentIds:
        if restoredIteration == 0:
            modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId})
        else:
            modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId,  'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)


    
    modelSaveFrequency=250
    modelMemorySize=10
    print ({'iterationIndex': restoredIteration, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
    for iterationIndex in range(restoredIteration + 1, numIterations):

        numCpuToUseWhileTrain = int(10)
        numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
        generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)

        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}

        trajecoriesNum=0
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

            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)

            #frequencyVersion: delete used model for disk space
            if iterationIndex % modelSaveFrequency != 0 and iterationIndex>=modelMemorySize:
                toDeleteNNModelPathParameters={'iterationIndex': iterationIndex-modelMemorySize, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
                toDeleteModelPathList = [generatetoDeleteNNModelPath(toDeleteNNModelPathParameters) for generatetoDeleteNNModelPath in generatetoDeleteNNModelPathList ]
                for toDeleteModelPath in toDeleteModelPathList:
                    # print('todelete:',toDeleteModelPath)
                    if os.path.isfile(toDeleteModelPath):
                        os.remove(toDeleteModelPath)
                        print('delete',toDeleteModelPath)
                    else:
                        print('no such file',toDeleteModelPath)
            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)


            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))


if __name__ == '__main__':
    # main()
    parameters={}
    parameters['numTrainStepEachIteration'] = 1
    parameters['numTrajectoriesPerIteration'] = 10
    iterateTrainOneCondition(parameters)