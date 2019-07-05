import sys
import os
src = os.path.join(os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import pandas as pd
import envNoPhysics as env
import state
import policyValueNet as net
import policies
from collections import OrderedDict
import pickle
import math
from episode import SampleTrajectory, chooseGreedyAction
from analyticGeometryFunctions import transitePolarToCartesian, \
    computeAngleBetweenVectors
from evaluationFunctions import GetSavePath, ComputeStatistics, \
    LoadMultipleTrajectoriesFile
import trainTools
from pylab import plt


class GenerateTrainedModel:

    def __init__(self, getSavePathForModel, getSavePathForTrajectory,
                 generateModel, generateTrain, generatePolicy, sampleTrajectory,
                 numTrials):
        self.getSavePathForModel = getSavePathForModel
        self.getSavePathForTrajectory = getSavePathForTrajectory
        self.generateTrain = generateTrain
        self.generateModel = generateModel
        self.generatePolicy = generatePolicy
        self.sampleTrajectory = sampleTrajectory
        self.numTrials = numTrials

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataType = df.index.get_level_values('trainingDataType')[0]
        batchSize = df.index.get_level_values('batchSize')[0]
        trainingStep = df.index.get_level_values('trainingStep')[0]
        neuronsPerLayer = df.index.get_level_values('neuronsPerLayer')[0]
        sharedLayers = df.index.get_level_values('sharedLayers')[0]
        actionLayers = df.index.get_level_values('actionLayers')[0]
        valueLayers = df.index.get_level_values('valueLayers')[0]
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
        model = self.generateModel([neuronsPerLayer] * sharedLayers,
                                   [neuronsPerLayer] * actionLayers,
                                   [neuronsPerLayer] * valueLayers)
        if os.path.exists(saveModelDir):
            trainedModel = net.restoreVariables(model, modelPath)
        else:
            train = self.generateTrain(trainingStep, batchSize)
            trainedModel = train(model, trainData)
            net.saveVariables(trainedModel, modelPath)
        saveTrajName = self.getSavePathForTrajectory(parameters)
        if os.path.exists(saveTrajName):
            print("Trajectory exists {}".format(saveTrajName))
            with open(saveTrajName, "rb") as f:
                trajectories = pickle.load(f)
        else:
            sheepPolicy, wolfPolicy = self.generatePolicy(trainedModel)
            policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
            trajectories = [self.sampleTrajectory(policy) for _ in range(self.numTrials)]
            with open(saveTrajName, 'wb') as f:
                pickle.dump(trajectories, f)
        return pd.Series({"Trajectory": trajectories})


def main():
    dataDir = os.path.join(os.pardir, 'data', 'evaluateByEpisodeLength') #.. parentdir

    # sample trajectory
    sheepID = 0
    posIndex = [0, 1]
    getSheepPos = state.GetAgentPosFromState(sheepID, posIndex)
    wolfID = 1
    getWolfPos = state.GetAgentPosFromState(wolfID, posIndex)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    numOfAgent = 2
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)
    killZoneRadius = 25
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)
    maxRunningSteps = 30
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(
        xBoundary, yBoundary)
    transition = env.TransiteForNoPhysics(checkBoundaryAndAdjust)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transition, isTerminal,
                                        reset, chooseGreedyAction)

    # data set
    trainingDataDir = os.path.join(dataDir, "dataSets")
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
        print("No dataSet in: {}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)

    # NeuralNetwork Parameter
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    validationSize = 100
    trainTerminalController = trainTools.TrainTerminalController(
        lossHistorySize, lossChangeThreshold, validationSize)
    initActionCoefficient = (1, 1)
    initValueCoefficient = (1, 1)
    coefficientController = trainTools.CoefficientCotroller(
        initActionCoefficient, initValueCoefficient)
    reportInterval = 1000
    reporter = trainTools.TrainReporter(reportInterval)
    decayRate = 1
    decayStep = 1
    initLearningRate = 1e-4
    learningRateModifier = trainTools.LearningRateModifier(
        initLearningRate, decayRate, decayStep)
    testDataSize = 10000
    testDataType = 'actionDist'
    testData = [
        dataSet[varName][-testDataSize:]
        for varName in ['state', testDataType, 'value']
    ]
    generateTrain = lambda trainingStep, batchSize: net.Train(
        trainingStep, batchSize, net.sampleData, learningRateModifier,
        trainTerminalController, coefficientController, reporter, testData)
    numStateSpace = 4
    numActionSpace = 8
    regularizationFactor = 0
    valueRelativeErrBound = 0.1
    generateModel = net.GenerateModel(numStateSpace,
                                      numActionSpace,
                                      regularizationFactor,
                                      valueRelativeErrBound)

    degrees = [
        math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4, -math.pi * 3 / 4,
        -math.pi / 4, math.pi * 3 / 4
    ]
    establishAction = lambda speed, degree: tuple(
        np.round(speed * transitePolarToCartesian(degree)))
    sheepSpeed = 20
    sheepActionSpace = [
        establishAction(sheepSpeed, degree) for degree in degrees
    ]
    wolfSpeed = sheepSpeed * 0.95
    wolfActionSpace = [establishAction(wolfSpeed, degree) for degree in degrees]
    wolfPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    generateSheepPolicy = lambda trainedModel: net.ApproximatePolicy(
        trainedModel, sheepActionSpace)
    generatePolicy = lambda trainedModel: (generateSheepPolicy(trainedModel),
                                           wolfPolicy)

    # split & apply
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [10000, 30000, 60000]
    independentVariables['batchSize'] = [0, 1024, 4096]
    independentVariables['augmented'] = ['no']
    independentVariables['trainingStep'] = [1000, 5000, 10000, 50000]
    independentVariables['neuronsPerLayer'] = [64]
    independentVariables['sharedLayers'] = [3]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    extension = ".pickle"
    fixedParameter = {'learningRate': 1e-4}
    trainedModelOutputPath = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelOutputPath):
        os.mkdir(trainedModelOutputPath)
    getSavePathForModel = GetSavePath(trainedModelOutputPath, "",
                                      fixedParameter)
    evaluationTrajectoryOutputPath = os.path.join(dataDir, "trajectories")
    if not os.path.exists(evaluationTrajectoryOutputPath):
        os.mkdir(evaluationTrajectoryOutputPath)
    getSavePathForTrajectory = GetSavePath(evaluationTrajectoryOutputPath,
                                           extension, fixedParameter)

    numTrials = 1000
    generateTrainingOutput = GenerateTrainedModel(getSavePathForModel,
                                                  getSavePathForTrajectory,
                                                  generateModel, generateTrain,
                                                  generatePolicy,
                                                  sampleTrajectory, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput,
                                                      dataSet)

    loadTrajectories = LoadMultipleTrajectoriesFile(getSavePathForTrajectory,
                                                    pickle.load)
    computeStatistic = ComputeStatistics(loadTrajectories, len)
    statDF = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    print(statDF)

    # draw
    xStatistic = "trainingStep"
    yStatistic = "mean"
    lineStatistic = "batchSize"
    subplotStatistic = "trainingDataSize"
    figsize = (12, 10)
    figure = plt.figure(figsize=figsize)
    subplotNum = len(statDF.groupby(subplotStatistic))
    numOfPlot = 1
    for subplotKey, subPlotDF in statDF.groupby(subplotStatistic):
        for linekey, lineDF in subPlotDF.groupby(lineStatistic):
            ax = figure.add_subplot(1, subplotNum, numOfPlot)
            plotDF = lineDF.reset_index()
            plotDF.plot(x=xStatistic,
                        y=yStatistic,
                        ax=ax,
                        label=linekey,
                        title="{}:{}".format(subplotStatistic, subplotKey))
        numOfPlot += 1
    plt.legend(loc='best')
    plt.suptitle("{} vs {} episode length".format(xStatistic, yStatistic))
    figureName = "effect_{}_on_NNPerformance.png".format(xStatistic)
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == "__main__":
    main()
