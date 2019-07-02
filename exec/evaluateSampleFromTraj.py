import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import numpy as np
import pandas as pd
import envNoPhysics as env
import wrappers
import policyValueNet as net
import policies
import os
import play
from collections import OrderedDict
import pickle
import math
from analyticGeometryFunctions import transitePolarToCartesian, \
    computeAngleBetweenVectors
from evaluationFunctions import GetSavePath, ComputeStatistics, LoadTrajectories
import trainTools
from pylab import plt
tfseed=128


class GenerateTrainedModel:
    def __init__(self, getSavePathForModel, getSavePathForTrajectory, model,
                 generateTrain, generatePolicy, generateSample,
                 sampleTrajectory, numTrials):
        self.getSavePathForModel = getSavePathForModel
        self.getSavePathForTrajectory = getSavePathForTrajectory
        self.generateTrain = generateTrain
        self.generateSample = generateSample
        self.model = model
        self.generatePolicy = generatePolicy
        self.play = sampleTrajectory
        self.numTrials = numTrials

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trajNum, stepNum = df.index.get_level_values('batchSize')[0]
        preference = df.index.get_level_values('preference')[0]
        trainData = dataSet[:trainDataSize]
        indexLevelNames = df.index.names
        parameters = {levelName: df.index.get_level_values(levelName)[0] for
                      levelName in indexLevelNames}
        saveModelDir = self.getSavePathForModel(parameters)
        sortedParameters = sorted(parameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for
                                parameter in sortedParameters]
        modelName = '_'.join(nameValueStringPairs).replace(" ", "")
        modelPath = os.path.join(saveModelDir, modelName)
        if os.path.exists(saveModelDir):
            trainedModel = net.restoreVariables(self.model, modelPath)
        else:
            sample = self.generateSample(preference, trajNum, stepNum)
            train = self.generateTrain(sample)
            trainedModel = train(self.model, trainData)
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
    def __init__(self, xVariableName, yVaraibleName, lineVariable):
        self.xName = xVariableName
        self.yName = yVaraibleName
        self.lineName = lineVariable

    def __call__(self, df):
        fig, ax = plt.subplots()
        for key, subDF in df.groupby(self.lineName):
            plotDF = subDF.reset_index()
            plotDF.plot(x=self.xName, y=self.yName, ax=ax, label=key)
        plt.xlabel(self.xName)
        plt.ylabel(self.yName)
        plt.legend(loc='best')
        plt.title("{} vs {} episode length".format(self.xName, self.yName))


def getTestData(rawData, stateIndex, distIndex, valueIndex, dataID):
    points = [step for traj in rawData for step in traj]
    flattenPoints = [[np.array(point[stateIndex]).flatten(),
                      list(point[distIndex][dataID].values()),
                      point[valueIndex]] for point in points]
    processedData = zip(*flattenPoints)
    return processedData

def main():
    dataDir = "../data/evaluateSampleFromTraj"

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
                                                        wolfDriectChasingPolicy(state)])

    reset = env.Reset(xBoundary, yBoundary, numOfAgent)
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    # Train Models
    trainingDataDir = os.path.join(dataDir, "trainingTrajectories")
    dataSetName = "cBase=100_initPos=Random_maxRunningSteps=30_numSimulations=200_numTrajs=2500_rolloutSteps=10_withReward.pickle"
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
    fixedParameters['iteration'] = 800
    fixedParameters['reportInterval'] = 100
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
    fixedParameters['testDataType'] = 'actionDist'

    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [1000, 1500]
    independentVariables['batchSize'] = [(100, 1), (50, 2), (20, 5)]
    independentVariables['augmented'] = ['no']
    independentVariables['preference'] = [False]

    # generate NN
    trainTerminalController = trainTools.TrainTerminalController(fixedParameters['lossHistorySize']
                                                                 , fixedParameters['lossChangeThreshold'], fixedParameters['validationSize'])
    coefficientController = trainTools.CoefficientCotroller(fixedParameters['initActionCoefficient']
                                                            , fixedParameters['initValueCoefficient'])
    trainReporter = trainTools.TrainReporter(fixedParameters['iteration']
                                             , fixedParameters['reportInterval'])
    lrModifier = trainTools.LearningRateModifier(fixedParameters['learningRate'], fixedParameters['decayRate']
                                                 , fixedParameters['decayStep'])
    trajStateIndex = 0
    trajActionDistIndex = 2
    trajValueIndex = 3
    validationData= getTestData(dataSet[-fixedParameters['validationSize']:],
                           trajStateIndex, trajActionDistIndex,
                           trajValueIndex, sheepID)
    generateSample = lambda preference, trajNum, stepNum: \
        trainTools.SampleBatchFromTrajectory(sheepID, preference, trajStateIndex,
                                             trajActionDistIndex, trajValueIndex,
                                             trajNum, stepNum)
    generateTrain = lambda sample: net.Train(fixedParameters['iteration'], 1,
                                             sample, lrModifier,
                                            trainTerminalController,
                                             coefficientController,
                                             trainReporter, validationData)
    generateModel = net.GenerateModel(fixedParameters['numStateSpace'],
                                    fixedParameters['numActionSpace'],
                                    fixedParameters['regularizationFactor'],
                                    fixedParameters['valueRelativeErrBound'],
                                    seed=tfseed)
    model = generateModel([fixedParameters['neuronsPerLayer']*fixedParameters['sharedLayers']],
                          [fixedParameters['neuronsPerLayer']]*fixedParameters['actionLayers'],
                          [fixedParameters['neuronsPerLayer']]*fixedParameters['valueLayers'])
    generatePolicy = lambda trainedModel: net.ApproximatePolicy(trainedModel, sheepActionSpace)

    # sample trajectories
    maxRunningSteps = 30
    distoAction = lambda actionDist: play.worldDistToAction(play.agentDistToGreedyAction, actionDist)
    sampleTrajectory = play.SampleTrajectory(maxRunningSteps, sheepTransition, isTerminal, reset, distoAction)

    # random policy performance
    # randomPolicy = lambda state: sheepActionSpace[np.random.choice(range(8))]
    # trajectories = [len(sampleTrajectory(randomPolicy)) for _ in range(1000)]
    # print(sum(trajectories)/1000)

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
    generateTrainingOutput = GenerateTrainedModel(getSavePathForModel, getSavePathForTrajectory, model, generateTrain,
                                                  generatePolicy, generateSample, sampleTrajectory, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput, dataSet)

    loadTrajectories = LoadTrajectories(getSavePathForTrajectory)
    computeStatistic = ComputeStatistics(loadTrajectories, len)
    statDF = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    print(statDF)

    xVariableName = "trainingDataSize"
    yVaraibleName = "mean"
    lineVariable = ["batchSize"]
    drawer = DrawStatistic(xVariableName, yVaraibleName, lineVariable)
    drawer(statDF)
    figureName = "effect_{}_on_NNPerformance.png".format(xVariableName)
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == "__main__":
    main()