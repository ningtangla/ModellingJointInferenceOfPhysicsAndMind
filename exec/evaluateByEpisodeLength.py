import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import numpy as np
import pandas as pd
import envNoPhysics as env
import wrapperFunctions
import policyValueNet as net
import policies
import os
import play
from collections import OrderedDict
import pickle
import math
from analyticGeometryFunctions import transitePolarToCartesian, \
    computeAngleBetweenVectors, computeVectorNorm
from evaluationFunctions import GetSavePath, ComputeStatistics, LoadTrajectories
import trainTools
from pylab import plt
tfseed=128


class GenerateTrainedModel:
    def __init__(self, getSavePathForModel, getSavePathForTrajectory, model,
                 train, generatePolicy, sampleTrajectory, numTrials):
        self.getSavePathForModel = getSavePathForModel
        self.getSavePathForTrajectory = getSavePathForTrajectory
        self.train = train
        self.model = model
        self.generatePolicy = generatePolicy
        self.play = sampleTrajectory
        self.numTrials = numTrials

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataType = df.index.get_level_values('trainingDataType')[0]
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
        if os.path.exists(saveModelDir): # To Do, check if it works for model names
            trainedModel = net.restoreVariables(self.model, modelPath)
        else:
            trainedModel = self.train(self.model, trainData)
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


class DrawStatistic:
    def __init__(self, xVariableName, yVaraibleName):
        self.xName = xVariableName
        self.yName = yVaraibleName

    def __call__(self, df):
        plt.xlabel(self.xName)
        plt.ylabel(self.yName)
        plt.title("{} vs {} episode length".format(self.xName, self.yName))
        plotDF = df.reset_index()
        plotDF.plot(x=self.xName, y=self.yName)


def main():
    dataDir = "../data/evaluateByEpisodeLength"

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

    getSheepPos = wrapperFunctions.GetAgentPosFromState(sheepID, posIndex)
    getWolfPos = wrapperFunctions.GetAgentPosFromState(wolfID, posIndex)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary,
                                                                 yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    transition = env.TransiteForNoPhysics(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state),
                                                       [np.array(action),
                                                        wolfDriectChasingPolicy(state)])

    reset = env.Reset(xBoundary, yBoundary, numOfAgent)
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius, computeVectorNorm)

    # Train Models
    trainingDataDir = os.path.join(dataDir, "dataSets")
    dataSetName = "initPos=[30,30,20,20]_maxRunningSteps=30_numDataPoints=5800_numSimulations=200_numTrajs=200_rolloutSteps=10_standardizedReward=True.pickle"
    dataSetPath = os.path.join(trainingDataDir, dataSetName)
    if not os.path.exists(dataSetPath):
        print("No dataSet in:\n{}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)

    fixedParameters = OrderedDict()
    fixedParameters['numStateSpace'] = 4
    fixedParameters['numActionSpace'] = 8
    fixedParameters['learningRate'] = 1e-4
    fixedParameters['regularizationFactor'] = 0
    fixedParameters['valueRelativeErrBound'] = 0.1
    fixedParameters['iteration'] = 100000
    fixedParameters['batchSize'] = 0
    fixedParameters['reportInterval'] = 1000
    fixedParameters['lossChangeThreshold'] = 1e-8
    fixedParameters['lossHistorySize'] = 10
    fixedParameters['initActionCoefficient'] = (1, 1)
    fixedParameters['initValueCoefficient'] = (1, 1)
    fixedParameters['neuronsPerLayer'] = 64
    fixedParameters['netLayers'] = 4
    fixedParameters['decayRate'] = 1
    fixedParameters['decayStep'] = 1

    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [100, 200, 300]

    # generate NN
    trainTerminalController = trainTools.TrainTerminalController(fixedParameters['lossHistorySize']
                                                                 , fixedParameters['lossChangeThreshold'])
    coefficientController = trainTools.coefficientCotroller(fixedParameters['initActionCoefficient']
                                                            , fixedParameters['initValueCoefficient'])
    trainReporter = trainTools.TrainReporter(fixedParameters['iteration']
                                             , fixedParameters['reportInterval'])
    lrModifier = trainTools.learningRateModifier(fixedParameters['learningRate'], fixedParameters['decayRate']
                                                 , fixedParameters['decayStep'])
    train =net.Train(fixedParameters['iteration'], fixedParameters['batchSize'], lrModifier
                                                 , trainTerminalController, coefficientController, trainReporter)
    generateModel = net.GenerateModelSeparateLastLayer(fixedParameters['numStateSpace'],
                                                       fixedParameters['numActionSpace'],
                                                       fixedParameters['regularizationFactor'],
                                                       fixedParameters['valueRelativeErrBound'],
                                                       seed=tfseed)
    model = generateModel([fixedParameters['neuronsPerLayer']] * fixedParameters['netLayers'])
    generatePolicy = lambda trainedModel: net.ApproximatePolicy(trainedModel, sheepActionSpace)

    # sample trajectories
    maxRunningSteps = 100
    sampleTrajectory = play.SampleTrajectory(maxRunningSteps, sheepTransition, isTerminal, reset)

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

    numTrials = 100
    generateTrainingOutput = GenerateTrainedModel(getSavePathForModel, getSavePathForTrajectory, model, train,
                                                  generatePolicy, sampleTrajectory, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput, dataSet)

    loadTrajectories = LoadTrajectories(getSavePathForTrajectory)
    computeStatistic = ComputeStatistics(loadTrajectories, numTrials, len)
    statDF = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    print(statDF)

    xVariableName = "trainingDataSize"
    yVaraibleName = "mean"
    drawer = DrawStatistic(xVariableName, yVaraibleName)
    drawer(statDF)
    figureName = "effect_trainingDataSize_on_NNPerformance.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == "__main__":
    main()