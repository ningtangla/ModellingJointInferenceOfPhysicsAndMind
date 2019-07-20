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
    ApproximateActionPrior, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories, ActionToOneHot
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


class GetMcts():
    def __init__(self, actionSpace, numSimulations, selectChild, isTerminal, transit, terminalRewardList):
        self.numSimulations = numSimulations
        self.selectChild = selectChild
        self.actionSpace = actionSpace
        self.isTerminal = isTerminal
        self.transit = transit
        self.numActionSpace = len(actionSpace)
        self.terminalRewardList = terminalRewardList

    def __call__(self, agentId, NNModel, othersPolicy):
        if agentId == 0:
            transitInMCTS = lambda state, selfAction: self.transit(state, [selfAction, othersPolicy(state)])
        if agentId == 1:
            transitInMCTS = lambda state, selfAction: self.transit(state, [othersPolicy(state), selfAction])

        getApproximateActionPrior = ApproximateActionPrior(NNModel, self.actionSpace)
        getInitializeChildrenNNPrior = InitializeChildren(self.actionSpace, transitInMCTS, getApproximateActionPrior)

        getExpandNNPrior = Expand(self.isTerminal, getInitializeChildrenNNPrior)
        getStateFromNode = lambda node: list(node.id.values())[0]
        terminalReward = self.terminalRewardList[agentId]
        getEstimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, getStateFromNode, ApproximateValueFunction(NNModel))

        getMCTSNNPriorValue = MCTS(self.numSimulations, self.selectChild, getExpandNNPrior,
                                   getEstimateValue, backup, establishPlainActionDist)

        return getMCTSNNPriorValue


class PreparePolicyOneAgentGuideMCTS:
    def __init__(self, getMCTS, approximatePolicy):
        self.getMCTS = getMCTS
        self.approximatePolicy = approximatePolicy

    def __call__(self, agentId, multiAgentNNmodel):
        multiAgnetPolicyList = [self.approximatePolicy(NNModel) for NNModel in multiAgentNNmodel]
        othersPolicy = multiAgnetPolicyList.copy()
        del othersPolicy[agentId]
        selfNNmodel = multiAgentNNmodel[agentId]
        multiAgnetPolicyList[agentId] = self.getMCTS(agentId, selfNNmodel, othersPolicy)

        return multiAgnetPolicyList


class TrainOneAgent:
    def __init__(self, numTrajectoriesPerIteration, trajectoriesForTrainSaveDirectoryList, trajectorySaveExtension, actionToOneHot,
                 removeTerminalTupleFromTrajectory, addValuesToTrajectoryList, learningThresholdFactor, miniBatchSize, saveToBuffer,
                 sampleBatchFromBuffer, trainNN, nnModelSaveDirectoryList, killzoneRadius, sampleTrajectory):
        self.numTrajectoriesPerIteration = numTrajectoriesPerIteration
        self.trajectoriesForTrainSaveDirectoryList = trajectoriesForTrainSaveDirectoryList
        self.trajectorySaveExtension = trajectorySaveExtension
        self.actionToOneHot = actionToOneHot
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.addValuesToTrajectoryList = addValuesToTrajectoryList
        self.learningThresholdFactor = learningThresholdFactor
        self.miniBatchSize = miniBatchSize
        self.saveToBuffer = saveToBuffer
        self.sampleBatchFromBuffer = sampleBatchFromBuffer
        self.trainNN = trainNN
        self.nnModelSaveDirectoryList = nnModelSaveDirectoryList
        self.killzoneRadius = killzoneRadius
        self.sampleTrajectory = sampleTrajectory

    def __call__(self, agentId, multiAgentPolicyList, NNModel, replayBuffer, pathParametersAtIteration):
        multiAgentPolicy = lambda state: multiAgentPolicyList
        trajectoriesForAgentTrain = [self.sampleTrajectory(multiAgentPolicy) for _ in range(self.numTrajectoriesPerIteration)]
        trajectoriesForAgentTrainSaveDirectory = self.trajectoriesForTrainSaveDirectoryList[agentId]

        getTrajectorySavePath = GetSavePath(trajectoriesForAgentTrainSaveDirectory, self.trajectorySaveExtension, pathParametersAtIteration)
        dataSetPath = getTrajectorySavePath(pathParametersAtIteration)

        saveToPickle(trajectoriesForAgentTrain, dataSetPath)

        processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(self.actionToOneHot, sheepId)

        addValuesToTrajectory = addValuesToTrajectoryList[agentId]
        preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, self.removeTerminalTupleFromTrajectory, processTrajectoryForNN)

        processedTrajectories = preProcessTrajectories(trajectoriesForAgentTrain)
        updatedReplayBuffer = self.saveToBuffer(replayBuffer, processedTrajectories)

        if len(updatedReplayBuffer) >= self.learningThresholdFactor * self.miniBatchSize:
            wolfSampledBatch = self.sampleBatchFromBuffer(updatedReplayBuffer)
            trainData = [list(varBatch) for varBatch in zip(*wolfSampledBatch)]
            updatedWolfNNModel = self.trainNN(NNModel, trainData)
            NNModel = updatedWolfNNModel

        replayBuffer = updatedReplayBuffer
        nnModelDirectory = self.nnModelSaveDirectoryList[agentId]
        getModelSavePath = GetSavePath(
            nnModelDirectory, NNModelSaveExtension, pathParametersAtIteration)
        nnModelSavePaths = getModelSavePath({'killzoneRadius': self.killzoneRadius})
        savedVariablesWolf = saveVariables(NNModel, nnModelSavePaths)

        return NNModel, replayBuffer


def main():
    # Mujoco environment
    dirName = os.path.dirname(__file__)
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
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    maxRunningSteps = 25
    sheepAliveBonus = 1 / maxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus

    sheepTerminalPenalty = -1
    wolfTerminalReward = 1
    terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    sheepPlayReward = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    wolfPlayReward = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)

    decay = 1
    sheepAccumulateRewards = AccumulateRewards(decay, sheepPlayReward)
    wolfAccumulateRewards = AccumulateRewards(decay, wolfPlayReward)

    addValuesToSheepTrajectory = AddValuesToTrajectory(sheepAccumulateRewards)
    addValuesToWolfTrajectory = AddValuesToTrajectory(wolfAccumulateRewards)

    addValuesToTrajectoryList = [addValuesToSheepTrajectory, addValuesToWolfTrajectory]
    # sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    # wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    # numActionSpace = len(sheepActionSpace)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # NNGuidedMCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    numSimulations = 200  # 200
    getMCTS = GetMcts(actionSpace, numSimulations, selectChild, isTerminal, transit, terminalRewardList)

    # sample trajectory
    # maxRunningSteps = 25
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

    actionToOneHot = ActionToOneHot(actionSpace)

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
    trainNN = Train(numTrainStepsPerIteration, miniBatchSize, sampleData,
                    learningRateModifier(learningRate),
                    terminalController, coefficientController,
                    trainReporter)

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

    trajectoriesForTrainSaveDirectoryList = [trajectoriesForSheepTrainSaveDirectory, trajectoriesForWolfTrainSaveDirectory]
    nnModelSaveDirectoryList = [sheepNNModelSaveDirectory, wolfNNModelSaveDirectory]

# load wolf baseline for init iteration
    # wolfBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data','SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')
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
    # sheepBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data','SheepWolfBaselinePolicy', 'sheepBaselineNNPolicy')
    # baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
    #                           'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
    #                           'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}
    # getSheepBaselineModelSavePath = GetSavePath(sheepBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)
    # baselineModelTrainSteps = 1000
    # sheepBaselineNNModelSavePath = getSheepBaselineModelSavePath({'trainSteps': baselineModelTrainSteps})
    # sheepBaselienModel = restoreVariables(initializedNNModel, sheepBaselineNNModelSavePath)
    # sheepNNModel = sheepBaselienModel
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    replayBuffer = []
    numTrajectoriesPerIteration = 1
    learningThresholdFactor = 4
    numIterations = 2000
    startTime = time.time()

    approximatePolicy = lambda NNmodel: ApproximatePolicy(NNmodel, actionSpace)  # need fix import
    preparePolicyOneAgentGuideMCTS = PreparePolicyOneAgentGuideMCTS(getMCTS, approximatePolicy)
    trainOneAgent = TrainOneAgent(numTrajectoriesPerIteration, trajectoriesForTrainSaveDirectoryList, trajectorySaveExtension, actionToOneHot, removeTerminalTupleFromTrajectory, addValuesToTrajectoryList, learningThresholdFactor, miniBatchSize, saveToBuffer, sampleBatchFromBuffer, trainNN, nnModelSaveDirectoryList, killzoneRadius, sampleTrajectory)

    for iterationIndex in range(numIterations):
        print("ITERATION INDEX: ", iterationIndex)
        pathParametersAtIteration = generatePathParametersAtIteration(iterationIndex)
        multiAgentNNmodel = [sheepNNModel, wolfNNModel]

        multiAgentPolicyForSheepTrain = preparePolicyOneAgentGuideMCTS(sheepId, multiAgentNNmodel)
        updatedSheepNNModel, replayBuffer = trainOneAgent(sheepId, multiAgentPolicyForSheepTrain, sheepNNModel, replayBuffer, pathParametersAtIteration)

        multiAgentNNmodel = [updatedSheepNNModel, wolfNNModel]
        multiAgentPolicyForWolfTrain = preparePolicyOneAgentGuideMCTS(wolfId, multiAgentNNmodel)
        updatedWolfNNModel, replayBuffer = trainOneAgent(wolfId, multiAgentPolicyForWolfTrain, wolfNNModel, replayBuffer, pathParametersAtIteration)

        sheepNNModel = updatedSheepNNModel
        wolfNNModel = updatedWolfNNModel

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        self.numIterations, (endTime - startTime)))


# sheep play
#         approximateWolfPolicy = ApproximatePolicy(wolfNNModel, wolfActionSpace)
#         sheepPolicy = getSheepMCTS(sheepId, sheepNNModel, approximateWolfPolicy)
#         wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}
#         policyForSheepTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

#         trajectoriesForSheepTrain = [sampleTrajectory(policyForSheepTrain) for _ in range(numTrajectoriesPerIteration)]
#         getSheepTrajectorySavePath = GetSavePath(
#             trajectoriesForSheepTrainSaveDirectory, trajectorySaveExtension, pathParametersAtIteration)

#         sheepDataSetPath = getSheepTrajectorySavePath(pathParametersAtIteration)
#         saveToPickle(trajectoriesForSheepTrain, sheepDataSetPath)

#         # sheepDataSetTrajectories = loadFromPickle(sheepDataSetPath)
#         sheepDataSetTrajectories = trajectoriesForSheepTrain

# # sheep train
#         processedSheepTrajectories = preProcessSheepTrajectories(sheepDataSetTrajectories)
#         updatedReplayBuffer = saveToBuffer(replayBuffer, processedSheepTrajectories)
#         if len(updatedReplayBuffer) >= learningThresholdFactor * miniBatchSize:
#             sheepSampledBatch = sampleBatchFromBuffer(updatedReplayBuffer)
#             sheepTrainData = [list(varBatch) for varBatch in zip(*sheepSampledBatch)]
#             updatedSheepNNModel = trainNN(sheepNNModel, sheepTrainData)
#             sheepNNModel = updatedSheepNNModel

#         replayBuffer = updatedReplayBuffer

#         getSheepModelSavePath = GetSavePath(
#             sheepNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
#         sheepNNModelSavePaths = getSheepModelSavePath({'killzoneRadius': killzoneRadius})
#         savedVariablesSheep = saveVariables(sheepNNModel, sheepNNModelSavePaths)


# wolf play
#         approximateSheepPolicy = lambda state: (0, 0)  # static sheep
#         # approximateSheepPolicy = ApproximatePolicy(sheepNNModel, sheepActionSpace)
#         wolfPolicy = getWolfMCTS(wolfId, wolfNNModel, approximateSheepPolicy)
#         sheepPolicy = lambda state: {approximateSheepPolicy(state): 1}
#         policyForWolfTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

#         trajectoriesForWolfTrain = [sampleTrajectory(policyForWolfTrain) for _ in range(numTrajectoriesPerIteration)]

#         getWolfTrajectorySavePath = GetSavePath(
#             trajectoriesForWolfTrainSaveDirectory, trajectorySaveExtension, pathParametersAtIteration)
#         wolfDataSetPath = getWolfTrajectorySavePath(pathParametersAtIteration)
#         saveToPickle(trajectoriesForWolfTrain, wolfDataSetPath)

#         wolfDataSetTrajectories = loadFromPickle(wolfDataSetPath)

# # wolf train
#         processedWolfTrajectories = preProcessWolfTrajectories(wolfDataSetTrajectories)
#         updatedReplayBuffer = saveToBuffer(replayBuffer, processedWolfTrajectories)
#         if len(updatedReplayBuffer) >= learningThresholdFactor * miniBatchSize:
#             wolfSampledBatch = sampleBatchFromBuffer(updatedReplayBuffer)
#             wolfTrainData = [list(varBatch) for varBatch in zip(*wolfSampledBatch)]
#             updatedWolfNNModel = trainNN(wolfNNModel, wolfTrainData)
#             wolfNNModel = updatedWolfNNModel

#         replayBuffer = updatedReplayBuffer

#         getWolfModelSavePath = GetSavePath(
#             wolfNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
#         wolfNNModelSavePaths = getWolfModelSavePath({'killzoneRadius': killzoneRadius})
#         savedVariablesWolf = saveVariables(wolfNNModel, wolfNNModelSavePaths)

#     endTime = time.time()
#     print("Time taken for {} iterations: {} seconds".format(
#         self.numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()
