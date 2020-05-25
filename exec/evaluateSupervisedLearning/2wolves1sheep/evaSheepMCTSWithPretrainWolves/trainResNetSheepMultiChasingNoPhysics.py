import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, RewardFunctionWithWall
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


def drawPerformanceLine(dataDf, axForDraw, deth):
    for learningRate, grp in dataDf.groupby('learningRate'):
        grp.index = grp.index.droplevel('learningRate')
        grp.plot(ax=axForDraw, label='learningRate={}'.format(learningRate), y='actionLoss',
                 marker='o', logx=False)


class TrainModelForConditions:
    def __init__(self, trainIntervelIndexes, trainStepsIntervel, trainData, NNModel, getTrain, getModelSavePath):
        self.trainIntervelIndexes = trainIntervelIndexes
        self.trainStepsIntervel = trainStepsIntervel
        self.trainData = trainData
        self.NNModel = NNModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, parameters):
        print(parameters)
        miniBatchSize = parameters['miniBatchSize']
        learningRate = parameters['learningRate']
        depth = parameters['depth']

        model = self.NNModel
        train = self.getTrain(miniBatchSize, learningRate)
        parameters.update({'trainSteps': 0})
        modelSavePath = self.getModelSavePath(parameters)
        saveVariables(model, modelSavePath)

        for trainIntervelIndex in self.trainIntervelIndexes:
            parameters.update({'trainSteps': trainIntervelIndex * self.trainStepsIntervel})
            modelSavePath = self.getModelSavePath(parameters)
            if not os.path.isfile(modelSavePath + '.index'):
                trainedModel = train(model, self.trainData)
                saveVariables(trainedModel, modelSavePath)
            else:
                trainedModel = restoreVariables(model, modelSavePath)
            model = trainedModel


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = 256  # [64, 128, 256, 512]
    manipulatedVariables['learningRate'] = 1e-4  # [1e-2, 1e-3, 1e-4, 1e-5]
    manipulatedVariables['depth'] = 9  # [2 ,4, 6, 8]

    # productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    # parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trajectories')
    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    numSimulations = 100
    maxRunningSteps = 150
    killzoneRadius = 30  # 80
    sheepId = 0

    dataSetFixedParameters = {'agentId': sheepId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    dataSetExtension = '.pickle'
    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    numOfAgent = 3
    sheepId = 0
    wolf1Id = 1
    wolf2Id = 2
    xPosIndex = [0, 1]

    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolf1Pos = GetAgentPosFromState(wolf1Id, xPosIndex)
    getWolf2Pos = GetAgentPosFromState(wolf2Id, xPosIndex)

    playAliveBonus = 1 / maxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = killzoneRadius

    playIsTerminalByWolf1 = IsTerminal(playKillzoneRadius, getSheepPos, getWolf1Pos)
    playIsTerminalByWolf2 = IsTerminal(playKillzoneRadius, getSheepPos, getWolf2Pos)

    def playIsTerminal(state): return playIsTerminalByWolf1(state) or playIsTerminalByWolf2(state)

    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    # xBoundary = [0, 600]
    # yBoundary = [0, 600]
    # safeBound = 80
    # wallDisToCenter = xBoundary[-1] / 2
    # wallPunishRatio = 3
    # playReward = RewardFunctionWithWall(playAliveBonus, playDeathPenalty, safeBound, wallDisToCenter, wallPunishRatio, playIsTerminal, getSheepPos)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 2
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

    numActionSpace = len(sheepActionSpace)
    actionToOneHot = ActionToOneHot(sheepActionSpace)

    actionIndex = 1

    def getTerminalActionFromTrajectory(trajectory): return trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    print(len(loadedTrajectories))

    def filterState(timeStep): return (timeStep[0][0:numOfAgent], timeStep[1], timeStep[2])
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]

    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))
    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]

    # neural network init and save path
    numStateSpace = 2 * numOfAgent
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    depth = 9
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    sheepNNmodel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

    def getNNModel(depth): return generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)

    # function to train NN model
    terminalThreshold = 1e-10
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    # terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)

    def terminalController(evalDict, numSteps): return False
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 10000
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1

    def learningRateModifier(learningRate): return LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)

    def getTrainNN(batchSize, learningRate): return Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController, trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trainedResNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    trainIntervelIndexes = list(range(6))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, sheepNNmodel, getTrainNN, getNNModelSavePath)

    trainModelForConditions(manipulatedVariables)
    # train models for all conditions
    # numCpuCores = os.cpu_count()
    # print(numCpuCores)
    # numCpuToUse = int(0.8*numCpuCores)
    # trainPool = mp.Pool(numCpuToUse)
    # #trainedModels = [trainPool.apply_async(trainModelForConditions, (parameters,)) for parameters in parametersAllCondtion]
    # trainPool.map(trainModelForConditions, parametersAllCondtion)


if __name__ == '__main__':
    main()
