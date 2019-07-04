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
TFSEED=128


class GenerateTrainedModel:
    def __init__(self, getSavePathForModel, getSavePathForTrajectory, getModel,
                 generateTrain, generatePolicy, generateTrainReporter,
                 sampleTrajectory, numTrials):
        self.getSavePathForModel = getSavePathForModel
        self.getSavePathForTrajectory = getSavePathForTrajectory
        self.generateTrain = generateTrain
        self.generateTrainReporter = generateTrainReporter
        self.getModel =getModel
        self.generatePolicy = generatePolicy
        self.play = sampleTrajectory
        self.numTrials = numTrials

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataType = df.index.get_level_values('trainingDataType')[0]
        batchSize = df.index.get_level_values('batchSize')[0]
        trainingStep = df.index.get_level_values('trainingStep')[0]
        reporter = self.generateTrainReporter(trainingStep)
        trainData = [dataSet[varName][:trainDataSize] for varName in
                     ['state', trainDataType, 'value']]
        indexLevelNames = df.index.names
        parameters = {levelName: df.index.get_level_values(levelName)[0] for
                      levelName in indexLevelNames}
        saveModelDir = self.getSavePathForModel(parameters)
        sortedParameters = sorted(parameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for
                                parameter in sortedParameters]
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
            print("Traj exists {}".format(saveTrajName))
            with open(saveTrajName, "rb") as f:
                trajectories = pickle.load(f)
        else:
            nnPolicy = self.generatePolicy(trainedModel)
            trajectories = [self.play(nnPolicy) for _ in range(self.numTrials)]
            with open(saveTrajName, 'wb') as f:
                pickle.dump(trajectories, f)
        return pd.Series({"Trajectory": trajectories})


def main():
    dataDir = "../data/evaluateByEpisodeLength" #用os

    # env
    sheepID = 0
    wolfID = 1
    posIndex = [0, 1]
    numOfAgent = 2
    sheepSpeed = 20
    wolfSpeed = sheepSpeed * 0.95
    degrees = [math.pi/2,0,math.pi,-math.pi/2,
               math.pi/4,-math.pi*3/4,-math.pi/4,math.pi*3/4]
    sheepActionSpace = [tuple(np.round(sheepSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    wolfActionSpace = [tuple(np.round(wolfSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    print(sheepActionSpace)
    print(wolfActionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    killZoneRadius = 25

    getSheepPos = wrappers.GetAgentPosFromState(sheepID, posIndex)
    getWolfPos = wrappers.GetAgentPosFromState(wolfID, posIndex)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary,
                                                                 yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    transition = env.TransiteForNoPhysics(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state),
                                                       [np.array(action),
                                                        wolfDriectChasingPolicy(state)]) #refactor

    reset = env.Reset(xBoundary, yBoundary, numOfAgent)
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

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
    fixedParameters = OrderedDict()
    fixedParameters['numStateSpace'] = 4
    fixedParameters['numActionSpace'] = 8
    fixedParameters['regularizationFactor'] = 0
    fixedParameters['valueRelativeErrBound'] = 0.1
    fixedParameters['reportInterval'] = 1000
    fixedParameters['lossChangeThreshold'] = 1e-6
    fixedParameters['lossHistorySize'] = 10
    fixedParameters['initActionCoefficient'] = (1, 1)
    fixedParameters['initValueCoefficient'] = (1, 1)
    fixedParameters['neuronsPerLayer'] = 64
    fixedParameters['sharedLayers'] = 3
    fixedParameters['actionLayers'] = 1
    fixedParameters['valueLayers'] = 1
    fixedParameters['decayRate'] = 1
    fixedParameters['decayStep'] = 1
    fixedParameters['validationSize'] = 100
    fixedParameters['testDataSize'] = 10000
    fixedParameters['testDataType'] = 'actionDist'

    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [10000, 30000, 60000]
    independentVariables['batchSize'] = [0, 1024, 4096]
    independentVariables['augmented'] = ['yes']
    independentVariables['learningRate'] = [1e-4]
    independentVariables['trainingStep'] = [1000, 5000, 10000, 50000]
    print(independentVariables['augmented'])
    print(dataSetPath)

    # generate NN
    trainTerminalController = trainTools.TrainTerminalController(fixedParameters['lossHistorySize']
                                                                 , fixedParameters['lossChangeThreshold'], fixedParameters['validationSize'])
    coefficientController = trainTools.CoefficientCotroller(fixedParameters['initActionCoefficient']
                                                            , fixedParameters['initValueCoefficient'])
    generateTrainReporter = lambda trainingStep: trainTools.TrainReporter(trainingStep,
                                             fixedParameters['reportInterval'])
    lrModifier = trainTools.LearningRateModifier(independentVariables['learningRate'][0], fixedParameters['decayRate']
                                                 , fixedParameters['decayStep'])
    testData = [dataSet[varName][-fixedParameters['testDataSize']:] for varName in
                     ['state', fixedParameters['testDataType'], 'value']]
    generateTrain = lambda trainingStep, batchSize, trainReporter: net.Train(trainingStep, batchSize, net.sampleData, lrModifier
                                                 , trainTerminalController, coefficientController, trainReporter, testData)
    generateModel = net.GenerateModel(fixedParameters['numStateSpace'],
                                    fixedParameters['numActionSpace'],
                                    fixedParameters['regularizationFactor'],
                                    fixedParameters['valueRelativeErrBound'],
                                    seed=TFSEED)
    getModel = lambda: generateModel([fixedParameters['neuronsPerLayer']*fixedParameters['sharedLayers']],
                          [fixedParameters['neuronsPerLayer']]*fixedParameters['actionLayers'],
                          [fixedParameters['neuronsPerLayer']]*fixedParameters['valueLayers'])
    generatePolicy = lambda trainedModel: net.ApproximatePolicy(trainedModel, sheepActionSpace)

    # sample trajectories
    maxRunningSteps = 30
    distoAction = lambda actionDist: play.worldDistToAction(play.agentDistToGreedyAction, actionDist) #改
    sampleTrajectory = play.SampleTrajectory(maxRunningSteps, sheepTransition, isTerminal, reset, distoAction)

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
    getSavePathForTrajectory = GetSavePath(evaluationTrajectoryOutputPath, extension)

    numTrials = 1000
    generateTrainingOutput = GenerateTrainedModel(getSavePathForModel, getSavePathForTrajectory, getModel, generateTrain,
                                                  generatePolicy, generateTrainReporter,
                                                  sampleTrajectory, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput, dataSet)


    loadTrajectories = LoadTrajectories(getSavePathForTrajectory)
    computeStatistic = ComputeStatistics(loadTrajectories, len)
    statDF = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    with open(os.path.join(dataDir, "temp.pkl"), 'wb')as f:
        pickle.dump(statDF, f)
    print(statDF)

    xStatistic = "trainingStep"
    yStatistic = "mean"
    lineStatistic = "batchSize"
    subplotStatistic = "trainingDataSize"
    figure = plt.figure(figsize=(12, 10))
    subplotDFs = statDF.groupby(subplotStatistic)
    for index in range(len(subplotDFs)):
        subplotKey, subPlotDF = subplotDFs[index]
        for linekey, lineDF in subPlotDF.groupby(lineStatistic):
            ax = figure.add_subplots(len(subplotDFs), 1, index+1)
            plotDF = lineDF.reset_index()
            plotDF.plot(x=xStatistic, y=yStatistic, ax=ax, label=linekey)
            ax.title(subplotKey)
    plt.legend(loc='best')
    plt.suptitle("{} vs {} episode length".format(xStatistic, yStatistic))
    figureName = "effect_trainingDataSize_on_NNPerformance.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == "__main__":
    main()