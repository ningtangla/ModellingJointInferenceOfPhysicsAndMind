import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNetTemp import GenerateModelSeparateLastLayer, Train, saveVariables, ApproximatePolicy, \
    restoreVariables
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, Reset
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.neuralNetwork.trainToolsTemp import CoefficientController, TrainTerminalController, TrainReporter
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy, stationaryAgentPolicy
from src.play import SampleTrajectory, worldDistToAction, agentDistToGreedyAction
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromTrajectory, GetStateFromTrajectory, \
    GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.measure import ComputeOptimalNextPos, DistanceBetweenActualAndOptimalNextPosition
from exec.evaluationFunctions import LoadTrajectories, ComputeStatistics
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
from mujoco_py import load_model_from_path, MjSim


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


def saveData(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


def drawPerformanceLine(dataDf, axForDraw, miniBatchSize):
    for learningRate, grp in dataDf.groupby('learningRate'):
        grp = grp.droplevel('learningRate')
        grp.plot(ax=axForDraw, label='learningRate={}'.format(learningRate), y='mean',
                 title='miniBatchSize: {}'.format(miniBatchSize), marker='o', logx=True)


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index
                                   in range(len(self.actionSpace))])

        return oneHotAction


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot, addValuesToTrajectory):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot
        self.addValuesToTrajectory = addValuesToTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        allTimeStepTuples = [tup for trajectory in trajectoriesWithValues for tup in trajectory]
        tuplesFiltered = list(filter(lambda tup: tup[self.actionIndex] is not None, allTimeStepTuples))
        print("{} data points remain after filtering".format(len(tuplesFiltered)))
        tuplesProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
                           for state, actions, actionDist, value in tuplesFiltered]

        return tuplesProcessed


class GenerateModel:
    def __init__(self, hiddenWidths, getGeneratePolicyNet):
        self.hiddenWidths = hiddenWidths
        self.getGeneratePolicyNet = getGeneratePolicyNet

    def __call__(self, learningRate):
        generatePolicyNet = self.getGeneratePolicyNet(learningRate)
        model = generatePolicyNet(self.hiddenWidths)

        return model


class GetPolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, NNModel):
        sheepPolicy = self.getSheepPolicy(NNModel)
        wolfPolicy = self.getWolfPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class TrainModelForConditions:
    def __init__(self, trainData, generateModel, getTrain, getModelSavePath):
        self.trainData = trainData
        self.generateModel = generateModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, oneConditionDf):
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        modelSavePath = self.getModelSavePath(parameters)

        if not os.path.isfile(modelSavePath + '.index'):
            model = self.generateModel(learningRate)
            train = self.getTrain(trainSteps, miniBatchSize)
            trainedModel = train(model, self.trainData)
            saveVariables(trainedModel, modelSavePath)

        return None


class GenerateTrajectories:
    def __init__(self, numTrials, getPolicyFromParameters, getSampleTrajectory, getTrajectorySavePath, saveData):
        self.numTrials = numTrials
        self.getPolicyFromParameters = getPolicyFromParameters
        self.getSampleTrajectory = getSampleTrajectory
        self.getTrajectorySavePath = getTrajectorySavePath
        self.saveData = saveData

    def __call__(self, oneConditionDf):
        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        trajectorySavePath = self.getTrajectorySavePath(parameters)

        if not os.path.isfile(trajectorySavePath):
            policy = self.getPolicyFromParameters(parameters)
            allSampleTrajectories = [self.getSampleTrajectory(trial) for trial in range(self.numTrials)]
            allTrajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
            self.saveData(allTrajectories, trajectorySavePath)

        return None


def main():
    # important parameters
    evalNumTrials = 100  # 200

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = [16, 64, 256, 512, 10000]
    manipulatedVariables['learningRate'] = [1e-4, 1e-6, 1e-2]
    manipulatedVariables['trainSteps'] = [1, 10, 100, 1000, 10000]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'NNPolicyVaryHyperParametersSheepEscapeWolfMujoco',
                                    'trajectories', 'train')
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 10
    dataSetNumSimulations = 75
    dataSetNumTrials = 1000
    dataSetQPosInit = (0, 0, 0, 0)
    dataSetQPosInitNoise = 9.7
    dataSetSheepPolicyName = 'MCTS'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'sheepPolicyName': dataSetSheepPolicyName, 'qPosInitNoise': dataSetQPosInitNoise}
    dataSetPath = getDataSetPath(dataSetConditionVariables)

    dataSetTrajectories = loadData(dataSetPath)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 0.05
    playDeathPenalty = -1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot, addValuesToTrajectory)
    stateActionValueTriplesProcessed = preProcessTrajectories(dataSetTrajectories)

    # shuffle and separate states and actions
    random.shuffle(stateActionValueTriplesProcessed)
    trainData = [[state for state, action, value in stateActionValueTriplesProcessed],
                 [action for state, action, value in stateActionValueTriplesProcessed],
                 np.asarray([value for state, action, value in stateActionValueTriplesProcessed])]

    # initialise model for training
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    getGeneratePolicyNet = lambda learningRate: GenerateModelSeparateLastLayer(numStateSpace, numActionSpace,
                                                                               learningRate, regularizationFactor)
    generateModel = GenerateModel(hiddenWidths, getGeneratePolicyNet)

    # train models
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientController(initActionCoeff, initValueCoeff)
    reportInterval = 500
    getTrain = lambda trainSteps, batchSize: Train(trainSteps, batchSize, terminalController, coefficientController,
                                                   TrainReporter(trainSteps, reportInterval))

    # get path to save trained models
    modelFixedParameters = {'dataSetMaxRunningSteps': dataSetMaxRunningSteps, 'dataSetQPosInit': dataSetQPosInit,
                            'dataSetNumSimulations': dataSetNumSimulations, 'dataSetNumTrials': dataSetNumTrials,
                            'dataSetSheepPolicyName': dataSetSheepPolicyName}
    modelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'NNPolicyVaryHyperParametersSheepEscapeWolfMujoco',
                                      'trainedModels')

    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelFixedParameters)

    # function to train models
    trainModelForConditions = TrainModelForConditions(trainData, generateModel, getTrain, getModelSavePath)

    # train models for all conditions
    toSplitFrame.groupby(levelNames).apply(trainModelForConditions)

    # all agent policies
    getApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)
    actionMagnitude = 5
    heatSeekingDeterministicPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfPos, getSheepPos, actionMagnitude)
    getWolfPolicy = lambda NNModel: heatSeekingDeterministicPolicy
    getPolicy = GetPolicy(getApproximatePolicy, getWolfPolicy)

    # sample trajectory for evaluation
    evalMaxRunningSteps = 15
    dirName = os.path.dirname(__file__)
    evalEnvModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    evalModel = load_model_from_path(evalEnvModelPath)
    evalSimulation = MjSim(evalModel)
    evalKillzoneRadius = 0.5
    evalIsTerminal = IsTerminal(evalKillzoneRadius, getSheepPos, getWolfPos)
    evalNumSimulationFrames = 20
    transit = TransitionFunction(evalSimulation, evalIsTerminal, evalNumSimulationFrames)
    evalQVelInit = (0, 0, 0, 0)
    evalNumAgent = 2
    evalQPosInitNoise = 0
    evalQVelInitNoise = 0
    allEvalQPosInit = np.random.uniform(-9.7, 9.7, (evalNumTrials, 4))
    getResetFromQPosInit = lambda evalQPosInit: Reset(evalSimulation, evalQPosInit, evalQVelInit, evalNumAgent,
                                                      evalQPosInitNoise, evalQVelInitNoise)
    getQPosInitFromTrialIndex = lambda trialIndex: allEvalQPosInit[trialIndex]
    getResetFromTrialIndex = lambda trialIndex: getResetFromQPosInit(getQPosInitFromTrialIndex(trialIndex))
    distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    getSampleTrajectory = lambda trialIndex: SampleTrajectory(evalMaxRunningSteps, transit, evalIsTerminal,
                                                              getResetFromTrialIndex(trialIndex), distToAction)

    # get save path for trajectories
    sheepPolicyName = 'NN'
    trajectoryFixedParameters = {'maxRunningSteps': evalMaxRunningSteps, 'numTrials': evalNumTrials,
                                 'sheepPolicyName': sheepPolicyName}
    trajectorySaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'NNPolicyVaryHyperParametersSheepEscapeWolfMujoco',
                                           'trajectories', 'evaluate')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # function to generate trajectories for evaluation
    dummyLearningRate = 0
    modelToRestoreVariables = generateModel(dummyLearningRate)
    getModelFromPath = lambda path: restoreVariables(modelToRestoreVariables, path)
    getModelPathFromParameters = lambda parameters: getModelSavePath(parameters)
    getModelFromParameters = lambda parameters: getModelFromPath(getModelPathFromParameters(parameters))
    getPolicyFromParameters = lambda parameters: getPolicy(getModelFromParameters(parameters))

    generateTrajectories = GenerateTrajectories(evalNumTrials, getPolicyFromParameters, getSampleTrajectory,
                                                getTrajectorySavePath, saveData)

    # # generate trajectories for all conditions
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # measurement Function
    initTimeStep = 0
    stateIndex = 0
    getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepPos, getWolfPos,
                                                           computeAngleBetweenVectors)
    getOptimalAction = lambda state: agentDistToGreedyAction(optimalPolicy(state))
    stationaryAgentAction = lambda state: agentDistToGreedyAction(stationaryAgentPolicy(state))
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentAction(state)])
    computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction, sheepTransit,
                                                  getSheepPos)
    measurementTimeStep = 1
    getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepPos, getNextStateFromTrajectory)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos,
                                                                      getPosAtNextStepFromTrajectory)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadData)
    computeStatistics = ComputeStatistics(loadTrajectories, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print("statisticsDf")
    print(statisticsDf)

    # plot the results
    fig = plt.figure()
    numColumns = len(manipulatedVariables['miniBatchSize'])
    numRows = 1
    plotCounter = 1

    for miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylim(13, 15)
        # plt.ylabel('Distance between optimal and actual next position of sheep')
        drawPerformanceLine(grp, axForDraw, miniBatchSize)
        plotCounter += 1

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
