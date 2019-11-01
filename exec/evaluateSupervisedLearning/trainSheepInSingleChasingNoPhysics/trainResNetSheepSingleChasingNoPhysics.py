import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,RewardFunctionWithWall
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



class ProcessTrajectoryForPolicyValueNetSampleSteps:
    def __init__(self, actionToOneHot, agentId,sampleOneStepPerTraj):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId
        self.sampleOneStepPerTraj = sampleOneStepPerTraj

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)

        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        if  self.sampleOneStepPerTraj:
            choosenIndex = random.choice(range(len(processedTrajectory)))
            choosenStep = [processedTrajectory[choosenIndex]]
            return choosenStep
        else:
            return processedTrajectory


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

        model = self.NNModel
        train = self.getTrain(miniBatchSize, learningRate)
        parameters.update({'trainSteps': 0})
        modelSavePath = self.getModelSavePath(parameters)
        saveVariables(model, modelSavePath)

        for trainIntervelIndex in self.trainIntervelIndexes:
            parameters.update({'trainSteps': trainIntervelIndex*self.trainStepsIntervel})
            modelSavePath = self.getModelSavePath(parameters)
            if not os.path.isfile(modelSavePath + '.index'):
                trainedModel = train(model, self.trainData)
                saveVariables(trainedModel, modelSavePath)
            else:
                trainedModel = restoreVariables(model, modelSavePath)
            model = trainedModel

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['dataSize'] =  [5000]
    manipulatedVariables['depth'] =  [5]
    manipulatedVariables['sampleOneStepPerTraj'] =  [0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(trainOneCondition, parametersAllCondtion)

def trainOneCondition(parameters):
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = 256
    manipulatedVariables['learningRate'] =  1e-4

    depth =  int(parameters['depth'])
    dataSize = int(parameters['dataSize'])
    sampleOneStepPerTraj =  int(parameters['sampleOneStepPerTraj'])

    manipulatedVariables['depth'] = depth
    manipulatedVariables['dataSize'] = dataSize
    manipulatedVariables['sampleOneStepPerTraj'] =sampleOneStepPerTraj
    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    # dataSetDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'trajectoriesNoWallPunish')
    dataSetDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'trajectoriesStillAction')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 150
    dataSetNumSimulations = 100
    killzoneRadius = 30
    sheepId=0
    dataSetFixedParameters = {'agentId':sheepId,'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    numOfAgent = 2
    sheepId = 0
    wolf1Id = 1

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolf1Pos = GetAgentPosFromState(wolf1Id, xPosIndex)

    playAliveBonus = 1/dataSetMaxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = killzoneRadius #2

    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolf1Pos)


    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    # xBoundary = [0,600]
    # yBoundary = [0,600]
    # safeBound = 80
    # wallDisToCenter = xBoundary[-1]/2
    # wallPunishRatio = 3
    # playReward = RewardFunctionWithWall(playAliveBonus, playDeathPenalty, safeBound, wallDisToCenter, wallPunishRatio, playIsTerminal,getSheepPos)


    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepActionSpace.append((0,0))

    preyPowerRatio = 3
    actionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))

    numActionSpace = len(sheepActionSpace)


    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    # processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNetSampleSteps(actionToOneHot, sheepId, sampleOneStepPerTraj)

    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)

    loadedTrajectories = loadTrajectories(parameters={})[:dataSize]

    filterState = lambda timeStep: (timeStep[0][0:numOfAgent], timeStep[1], timeStep[2])#!!? magic
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    print(len(trajectories),parameters)

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


    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
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
    # terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    terminalController = lambda evalDict, numSteps: False
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 10000
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda batchSize, learningRate: Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController,trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(dirName, '..','..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trainedResNNModelsStillAction')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    trainIntervelIndexes = list(range(6))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, sheepNNModel, getTrainNN, getNNModelSavePath)

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
