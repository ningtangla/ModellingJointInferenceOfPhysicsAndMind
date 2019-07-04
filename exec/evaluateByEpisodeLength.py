import sys
import os
src = os.path.join('..', 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import pandas as pd
import envNoPhysics as env
import wrappers
import policyValueNet as net
import policies
import play
from collections import OrderedDict
import pickle
import math
from analyticGeometryFunctions import transitePolarToCartesian, \
    computeAngleBetweenVectors
from evaluationFunctions import GetSavePath, ComputeStatistics, LoadTrajectories
import trainTools
from pylab import plt
TFSEED = 128


class GenerateTrainedModel:

    def __init__(self, getSavePathForModel, getSavePathForTrajectory, getModel,
                 generateTrain, generatePolicy, generateTrainReporter,
                 sampleTrajectory, numTrials):
        self.getSavePathForModel = getSavePathForModel
        self.getSavePathForTrajectory = getSavePathForTrajectory
        self.generateTrain = generateTrain
        self.generateTrainReporter = generateTrainReporter
        self.getModel = getModel
        self.generatePolicy = generatePolicy
        self.play = sampleTrajectory
        self.numTrials = numTrials

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataType = df.index.get_level_values('trainingDataType')[0]
        batchSize = df.index.get_level_values('batchSize')[0]
        trainingStep = df.index.get_level_values('trainingStep')[0]
        reporter = self.generateTrainReporter(trainingStep)
        trainData = [
            dataSet[varName][:trainDataSize]
            for varName in ['state', trainDataType, 'value']
        ]
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        saveModelDir = self.getSavePathForModel(parameters)
        sortedParameters = sorted(parameters.items())
        nameValueStringPairs = [
            parameter[0] + '=' + str(parameter[1])
            for parameter in sortedParameters
        ]
        modelName = '_'.join(nameValueStringPairs).replace(" ", "")
        modelPath = os.path.join(saveModelDir, modelName)
        model = self.getModel()
        if os.path.exists(saveModelDir):
            trainedModel = net.restoreVariables(model, modelPath)
        else:
            train = self.generateTrain(trainingStep, batchSize, reporter)
            trainedModel = train(model, trainData)
            net.saveVariables(trainedModel, modelPath)
        saveTrajName = self.getSavePathForTrajectory(parameters)
        if os.path.exists(saveTrajName):
            # print("Traj exists {}".format(saveTrajName))
            with open(saveTrajName, "rb") as f:
                trajectories = pickle.load(f)
        else:
            nnPolicy = self.generatePolicy(trainedModel)
            trajectories = [self.play(nnPolicy) for _ in range(self.numTrials)]
            with open(saveTrajName, 'wb') as f:
                pickle.dump(trajectories, f)
        return pd.Series({"Trajectory": trajectories})


def main():
    dataDir = os.path.join('..', 'data', 'evaluateByEpisodeLength')

    # env
    sheepID = 0
    posIndex = [0, 1]
    getSheepPos = wrappers.GetAgentPosFromState(sheepID, posIndex)
    wolfID = 1
    getWolfPos = wrappers.GetAgentPosFromState(wolfID, posIndex)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(
        xBoundary, yBoundary)
    degrees = [
        math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4, -math.pi * 3 / 4,
        -math.pi / 4, math.pi * 3 / 4
    ]
    establishAction = lambda speed, degree: \
        tuple(np.round(speed * transitePolarToCartesian(degree)))
    sheepSpeed = 20
    sheepActionSpace = [
        establishAction(sheepSpeed, degree) for degree in degrees
    ]
    wolfSpeed = sheepSpeed * 0.95
    wolfActionSpace = [establishAction(wolfSpeed, degree) for degree in degrees]
    imaginedWolfAction = policies.HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    transition = env.TransiteForNoPhysics(checkBoundaryAndAdjust)
    sheepTransition = lambda state, sheepAction: transition(
        np.array(state), [np.array(sheepAction),
                          imaginedWolfAction(state)])
    numOfAgent = 2
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)
    killZoneRadius = 25
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)
    # sample trajectories
    maxRunningSteps = 30
    distoAction = lambda actionDist: play.worldDistToAction(
        play.agentDistToGreedyAction, actionDist)  #TODO: Refractor
    sampleTrajectory = play.SampleTrajectory(maxRunningSteps, sheepTransition,
                                             isTerminal, reset, distoAction)

    # Train Models
    trainingDataDir = os.path.join(dataDir, "augmentedDataSets")
    dataSetParameter = OrderedDict()
    dataSetParameter['cBase'] = 100
    dataSetParameter['initPos'] = 'Random'
    dataSetParameter['maxRunningSteps'] = 30
    dataSetParameter['numDataPoints'] = 68181
    dataSetParameter['numSimulations'] = 200
    dataSetParameter['numTrajs'] = 2500
    dataSetParameter['rolloutSteps'] = 10
    dataSetParameter['standardizedReward'] = 'True'
    dataSetExtension = '.pickle'
    getSavePathForDataSet = GetSavePath(trainingDataDir, dataSetExtension)
    dataSetPath = getSavePathForDataSet(dataSetParameter)
    if not os.path.exists(dataSetPath):
        print("No dataSet in:\n{}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)

    # 改回名称
    numStateSpace = 4
    numActionSpace = 8
    regularizationFactor = 0
    valueRelativeErrBound = 0.1
    reportInterval = 1000
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    initActionCoefficient = (1, 1)
    initValueCoefficient = (1, 1)
    neuronsPerLayer = 64
    sharedLayers = 3
    actionLayers = 1
    valueLayers = 1
    decayRate = 1
    decayStep = 1
    validationSize = 100
    testDataSize = 10000
    learningRate = 1e-4
    testDataType = 'actionDist'

    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [10000, 30000, 60000]
    independentVariables['batchSize'] = [0, 1024, 4096]
    independentVariables['augmented'] = ['no']
    independentVariables['learningRate'] = [learningRate]
    independentVariables['trainingStep'] = [1000, 5000, 10000, 50000]

    # generate NN
    trainTerminalController = trainTools.TrainTerminalController(
        lossHistorySize, lossChangeThreshold, validationSize)
    coefficientController = trainTools.CoefficientCotroller(
        initActionCoefficient, initValueCoefficient)
    generateTrainReporter = lambda trainingStep: trainTools.TrainReporter(
        trainingStep, reportInterval)
    lrModifier = trainTools.LearningRateModifier(learningRate, decayRate,
                                                 decayStep)
    testData = [
        dataSet[varName][-testDataSize:]
        for varName in ['state', testDataType, 'value']
    ]
    generateTrain = lambda trainingStep, batchSize, trainReporter: net.Train(
        trainingStep, batchSize, net.sampleData, lrModifier,
        trainTerminalController, coefficientController, trainReporter, testData)
    generateModel = net.GenerateModel(numStateSpace,
                                      numActionSpace,
                                      regularizationFactor,
                                      valueRelativeErrBound,
                                      seed=TFSEED)
    getModel = lambda: generateModel([neuronsPerLayer * sharedLayers],
                                     [neuronsPerLayer] * actionLayers,
                                     [neuronsPerLayer] * valueLayers)
    generatePolicy = lambda trainedModel: net.ApproximatePolicy(
        trainedModel, sheepActionSpace)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    extension = ".pickle"
    trainedModelOutputPath = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelOutputPath):
        os.mkdir(trainedModelOutputPath)
    getSavePathForModel = GetSavePath(trainedModelOutputPath, "")
    evaluationTrajectoryOutputPath = os.path.join(dataDir, "trajectories")
    if not os.path.exists(evaluationTrajectoryOutputPath):
        os.mkdir(evaluationTrajectoryOutputPath)
    getSavePathForTrajectory = GetSavePath(evaluationTrajectoryOutputPath,
                                           extension)

    numTrials = 1000
    generateTrainingOutput = GenerateTrainedModel(
        getSavePathForModel, getSavePathForTrajectory, getModel, generateTrain,
        generatePolicy, generateTrainReporter, sampleTrajectory, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput,
                                                      dataSet)

    loadTrajectories = LoadTrajectories(getSavePathForTrajectory)
    computeStatistic = ComputeStatistics(loadTrajectories, len)
    statDF = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    print(statDF)

    xStatistic = "trainingStep"
    yStatistic = "mean"
    lineStatistic = "batchSize"
    subplotStatistic = "trainingDataSize"
    figure = plt.figure(figsize=(12, 10))
    subplotDFs = statDF.groupby(subplotStatistic)
    numOfPlot = 1
    for subplotKey, subPlotDF in statDF.groupby(subplotStatistic):
        for linekey, lineDF in subPlotDF.groupby(lineStatistic):
            ax = figure.add_subplot(1, len(subplotDFs), numOfPlot)
            plotDF = lineDF.reset_index()
            plotDF.plot(x=xStatistic,
                        y=yStatistic,
                        ax=ax,
                        label=linekey,
                        title=subplotKey)
        numOfPlot += 1
    plt.legend(loc='best')
    plt.suptitle("{} vs {} episode length".format(xStatistic, yStatistic))
    figureName = "effect_trainingDataSize_on_NNPerformance.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == "__main__":
    main()
