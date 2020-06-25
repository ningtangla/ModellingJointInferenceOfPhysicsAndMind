import time
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform

from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.envMujocoRandomObstacles import SampleObscalesProperty, SetMujocoEnvXmlProperty, changeWallProperty,TransitionFunction,CheckAngentStackInWall,ResetUniformInEnvWithObstacles,getWallList
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, \
    LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData,  restoreVariables
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.NNGuidedMCTS import ApproximatePolicy,ApproximateValue


from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter,    LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel



def flattenStateInTrajectory(trajectory):
    flattenState =lambda state : np.array([i for s in state for i in s ]).reshape(1,-1)
    trajectoryWithFlattenState = [(flattenState(s), a, dist, v) for (s, a, dist, v) in trajectory]

    return trajectoryWithFlattenState

class PreprocessTrajectoriesForBuffer:
    def __init__(self, addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory,flattenStateInTrajectory):
        self.addMultiAgentValuesToTrajectory = addMultiAgentValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.flattenStateInTrajectory=flattenStateInTrajectory
    def __call__(self, trajectories):

        trajectoriesWithValues = [self.addMultiAgentValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        flattenedTrajectories = [self.flattenStateInTrajectory(trajectory) for trajectory in filteredTrajectories]

        return flattenedTrajectories


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


    # load save dir
    dirName = os.path.dirname(__file__)
    dataDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNNRandomObstacle')
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    #manupulate parameters
    restoredIteration = 106



    numAgents=2
    agentIds = list(range(numAgents))
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    maxRunningSteps = 30
    sheepAliveBonus = 1 / maxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus

    sheepTerminalPenalty = -1
    wolfTerminalReward = 1
    terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
    rewardMultiAgents = [rewardSheep, rewardWolf]

    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

    # neural network init
    numStateSpace = 20
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    depth = 9
    resBlock = 2
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths,resBlock) for agentId in agentIds]

    # replay buffer
    bufferSize = 2000
    saveToBuffer = SaveToBuffer(bufferSize)
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 256
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
    numTrainStepsPerIteration = 1
    trainReporter = TrainReporter(numTrainStepsPerIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRate = 0.0001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    trainNN = Train(numTrainStepsPerIteration, miniBatchSize, sampleData,learningRateModifier, terminalController, coefficientController,trainReporter)


    numSimulations = 200
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,
                       'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesSaveDirectory = os.path.join(dataDirectory, 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dataDirectory, 'ResNNModel')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)


    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    startTime = time.time()
    trainableAgentIds = [sheepId, wolfId]


    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory,removeTerminalTupleFromTrajectory,flattenStateInTrajectory)
    numTrajectoriesToStartTrain =4 * miniBatchSize
    numTrainStepEachIteration = 1
    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain,processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)

    for agentId in trainableAgentIds:
        modelPathBeforeTrain = generateNNModelSavePath({'iterationIndex': 0, 'agentId': agentId})
        saveVariables(multiAgentNNmodel[agentId], modelPathBeforeTrain)

    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMultiMCTSAgentTrajectoryObstacle.py'
    numCpuCores = os.cpu_count()
    numCpuToUse =int(0.8 * numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain,numCmdList)
    trajectoryBeforeTrainPathParamters = {'iterationIndex': 0}
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForParallel = LoadTrajectories(generateTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    # load trajectory function for trainBreak
    loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)

    # initRreplayBuffer
    replayBuffer = []

 
    if restoredIteration == 0:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

    # restore model
    for agentId in trainableAgentIds:
        modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)

    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={'depth':depth,'isResNet':0}, parametersWithSpecificValues={
        'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)


    #paralle sample trajectory
    numTrajectoriesPerIteration = 1
    numCpuToUseWhileTrain = int(16)
    numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
    sampleTrajectoryFileName = 'sampleMultiMCTSAgentTrajectoryRandomObstacle.py'
    generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)




    numIterations = 100000



    for iterationIndex in range(restoredIteration + 1, numIterations):
        # print("ITERATION INDEX: ", iterationIndex)
        # print("BUFFER SIZE: ", len(replayBuffer))

        # policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        # trajectories = [sampleTrajectory(policy) for _ in range(numTrajectoriesPerIteration)]
        # trajectoryPathParameters = {'iterationIndex': iterationIndex}
        # trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        # saveToPickle(trajectories, trajectorySavePath)

        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'depth':depth,'isResNet':0}

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
            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)

            saveVariables(updatedAgentNNModel, NNModelSavePath)

            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()