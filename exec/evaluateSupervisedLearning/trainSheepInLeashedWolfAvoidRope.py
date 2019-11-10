import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
from mujoco_py import load_model_from_path, MjSim
import itertools as it
import pathos.multiprocessing as mp

from src.constrainedChasingEscapingEnv.envMujoco import  TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
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

class IsTerminal:
    def __init__(self, minXDis, getSheepPos, getWolfPos, getRopeQPos):
        self.minXDis = minXDis
        self.getSheepPos = getSheepPos
        self.getWolfPos = getWolfPos
        self.getRopeQPos = getRopeQPos

    def __call__(self, state):
        state = np.asarray(state)
        posSheep = self.getSheepPos(state)
        posWolf = self.getWolfPos(state)
        posRope = [getPos(state) for getPos in self.getRopeQPos]
        L2NormDistanceForWolf = np.linalg.norm((posSheep - posWolf), ord=2)
        L2NormDistancesForRope = np.array([np.linalg.norm((posSheep - pos), ord=2) for pos in posRope])
        terminal = (L2NormDistanceForWolf <= self.minXDis) or np.any(L2NormDistancesForRope <= self.minXDis)

        return terminal


class TrainModelForConditions:
    def __init__(self, trainIntervelIndexes, trainStepsIntervel, trainData, getNNModel, getTrain, getModelSavePath):
        self.trainIntervelIndexes = trainIntervelIndexes
        self.trainStepsIntervel = trainStepsIntervel
        self.trainData = trainData
        self.getNNModel = getNNModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, parameters):
        print(parameters)
        miniBatchSize = parameters['miniBatchSize']
        learningRate = parameters['learningRate']
        depth = parameters['depth']

        model = self.getNNModel(depth)
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
    # important parameters
    sheepId = 0

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = [128, 256]
    manipulatedVariables['learningRate'] =  [1e-3, 1e-4]
    manipulatedVariables['depth'] = [ 4]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning',
                                    'sheepAvoidRopeTrajectories')
    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 25
    dataSetNumSimulations = 200
    killzoneRadius = 1

    dataSetFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    ropeIds = range(4,13)
    qPosIndex = [0, 1]
    getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
    getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
    getRopeQPos = [GetAgentPosFromState(ropeId, qPosIndex) for ropeId in ropeIds]
    isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos, getRopeQPos)
    
    playAliveBonus = 0.05
    playDeathPenalty = -1
    playKillzoneRadius = 1
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, isTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    preyPowerRatio = 0.7
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    numActionSpace = len(actionSpace)
    actionIndex = 1
    actionToOneHot = ActionToOneHot(sheepActionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    print(loadedTrajectories[0][0][0].shape)
    filterState = lambda timeStep: (timeStep[0][0:13], timeStep[1], timeStep[2])
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))
    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]

    selectedState = np.array(trainData[0])[:,:24]
    trainData[0] = selectedState

    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 24
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    getNNModel = lambda depth: generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)
    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)
    # import ipdb;ipdb.set_trace()
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

    NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    trainIntervelIndexes = list(range(11))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, getNNModel, getTrainNN, getNNModelSavePath)

    # train models for all conditions
    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.8*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    #trainedModels = [trainPool.apply_async(trainModelForConditions, (parameters,)) for parameters in parametersAllCondtion]
    trainPool.map(trainModelForConditions, parametersAllCondtion)

if __name__ == '__main__':
    main()
