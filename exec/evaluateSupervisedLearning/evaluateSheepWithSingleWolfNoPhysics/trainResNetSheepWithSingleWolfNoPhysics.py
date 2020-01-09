import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

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


def trainOneCondition(manipulatedVariables):
    depth = int(manipulatedVariables['depth'])
    dataSize = int(manipulatedVariables['dataSize'])

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trajectoriesWithStillAction')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 150
    dataSetNumSimulations = 100
    killzoneRadius = 30
    sheepId = 0
    dataSetFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    numOfAgent = 2
    sheepId = 0
    wolf1Id = 1

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolf1Pos = GetAgentPosFromState(wolf1Id, xPosIndex)

    playAliveBonus = 1 / dataSetMaxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = killzoneRadius

    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolf1Pos)

    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepActionSpace.append((0, 0))

    preyPowerRatio = 3
    actionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))

    numActionSpace = len(sheepActionSpace)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)

    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)

    trajectories = loadTrajectories(parameters={})[:dataSize]
    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))

    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]
    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]

    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    sheepNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

    initTimeStep = 0
    valueIndex = 3
    trainDataMeanAccumulatedReward = np.mean([tra[initTimeStep][valueIndex] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    # function to train NN model
    terminalThreshold = 1e-10
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 0
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 0
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    terminalController = lambda evalDict, numSteps: False
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 10000
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda batchSize, learningRate: Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController, trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trainedResNNModelsStillAction')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    numOfTrainStepsIntervel = 6
    trainIntervelIndexes = list(range(numOfTrainStepsIntervel))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, sheepNNModel, getTrainNN, getNNModelSavePath)
    trainModelForConditions(manipulatedVariables)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['dataSize'] = [5000]
    manipulatedVariables['depth'] = [5]
    manipulatedVariables['miniBatchSize'] = [256]
    manipulatedVariables['learningRate'] = [1e-4]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(trainOneCondition, parametersAllCondtion)


if __name__ == '__main__':
    main()