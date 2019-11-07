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

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
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
        dropOutRate = parameters['dropOutRate']
        # width = parameters['width']

        model = self.getNNModel(dropOutRate)
        train = self.getTrain(miniBatchSize, learningRate)
        parameters.update({'trainSteps': 0})
        modelSavePath = self.getModelSavePath(parameters)
        print(modelSavePath)
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

        trainedModel.close()
        model.close()

def main():
    # important parameters
    sheepId = 0

    # manipulated variables
    manipulatedVariables = OrderedDict()
<<<<<<< Updated upstream:exec/MultiAgentTrain/trainSheepResNNModelInVaryHyperParameters.py
=======
    manipulatedVariables['miniBatchSize'] = 256 #[64, 128, 256, 512]
    manipulatedVariables['learningRate'] =  1e-4#[1e-2, 1e-3, 1e-4, 1e-5]
    manipulatedVariables['depth'] = 5 #[2 ,4, 6, 8]
>>>>>>> Stashed changes:exec/evaluateSupervisedLearning/trainSheepInMultiChasingNoPhysics/trainMCTSSheepMultiChasingNoPhysics.py

    manipulatedVariables['miniBatchSize'] = [64, 128, 256]
    manipulatedVariables['learningRate'] =  [1e-3, 1e-4]
    manipulatedVariables['dropOutRate'] = [0, 0.2, 0.4]
    # manipulatedVariables['resBlock'] = [2 ,4, 6, 8]
    # manipulatedVariables['width'] = [32, 64 ,128, 256]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning',
                                    'trajectories')
    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
<<<<<<< Updated upstream:exec/MultiAgentTrain/trainSheepResNNModelInVaryHyperParameters.py
=======
    dataSetMaxRunningSteps = 150 #80
    dataSetNumSimulations = 200 #200
    killzoneRadius = 30 #2
>>>>>>> Stashed changes:exec/evaluateSupervisedLearning/trainSheepInMultiChasingNoPhysics/trainMCTSSheepMultiChasingNoPhysics.py

    dataSetMaxRunningSteps = 25
    dataSetNumSimulations = 100

    killzoneRadius = 2

    dataSetFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 0.05
    playDeathPenalty = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
<<<<<<< Updated upstream:exec/MultiAgentTrain/trainSheepResNNModelInVaryHyperParameters.py
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
=======
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
>>>>>>> Stashed changes:exec/evaluateSupervisedLearning/trainSheepInMultiChasingNoPhysics/trainMCTSSheepMultiChasingNoPhysics.py
    numActionSpace = len(actionSpace)
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    sheepActionSpace.append((0,0))

    predatorPowerRatio = 2
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

    actionIndex = 1
    actionToOneHot = ActionToOneHot(sheepActionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    trajectories = loadTrajectories(parameters={})
    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))
    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]

    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    # neural network init and save path
    numStateSpace = 12
    numActionSpace = 8
    regularizationFactor = 1e-2
    resBlockSize = 2
    initialization = 'uniform'
    nnStructure = ((256,) * 17, (256,), (256,))
    sharedWidths, actionLayerWidths, valueLayerWidths = nnStructure
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getNNModel = lambda  dropoutRate: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths, resBlockSize=resBlockSize,
                          initialization=initialization, dropoutRate=dropoutRate)

    # function to train NN model
    terminalThreshold = 1e-10
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    terminalController = lambda evalDict, numStep: False
    #terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)

    reportInterval = 50000
    trainStepsIntervel = 50000

    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda batchSize, learningRate: Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController,
                                                                     trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateResNN', 'trainedResSheepModels','res2depth17')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    trainIntervelIndexes = list(range(10,21))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, getNNModel, getTrainNN, getNNModelSavePath)

    # trainModelForConditions({'miniBatchSize':128,'learningRate':0.001,'dropOutRate':0.2})

    # train models for all conditions
    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.8*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    # trainedModels = [trainPool.apply_async(trainModelForConditions, (parameters,)) for parameters in parametersAllCondtion]
    models = trainPool.map(trainModelForConditions, parametersAllCondtion)

if __name__ == '__main__':
    main()