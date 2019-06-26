import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import pandas as pd
import os
import pickle
import policyValueNet as net
import trainTools
from collections import OrderedDict
from evaluationFunctions import GetSavePath
from pylab import plt


tfseed = 128


class GenerateTrainingLoss:
    def __init__(self, getSavePath, model, generateTrain, generateLRModifier):
        self.getSavePath = getSavePath
        self.generateLRModifier = generateLRModifier
        self.generateTrain = generateTrain
        self.model = model

    def __call__(self, df, dataSet):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataType = df.index.get_level_values('trainingDataType')[0]
        trainData = [dataSet[varName][:trainDataSize] for varName in ['state', trainDataType, 'value']]
        decayRate = df.index.get_level_values('decayRate')[0]
        decayStep = df.index.get_level_values('decayStep')[0]
        indexLevelNames = df.index.names
        parameters = {levelName: df.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)
        if os.path.exists(saveFileName):
            print("Data exists {}".format(saveFileName))
            with open(saveFileName, 'rb') as f:
                data = pickle.load(f)
            return pd.Series(data)

        lrModifier = self.generateLRModifier(decayRate, decayStep)
        train = self.generateTrain(lrModifier)
        trainedModel = train(self.model, trainData)
        evalTrain = net.evaluate(trainedModel, trainData)
        file = open(saveFileName, 'wb')
        pickle.dump(evalTrain, file)
        file.close()
        return pd.Series(evalTrain)


class GenerateDataFrameForDrawing:
    def __init__(self, xVariableName, yVariableName, subplotVariableName, lineVariableName):
        self.xVariableName = xVariableName
        self.yVariableName = yVariableName
        self.subplotVariableName = subplotVariableName
        self.lineVariableName = lineVariableName

    def __call__(self, df):
        df = df[self.yVariableName]
        figure = plt.figure(figsize=(12, 10))
        numplot = 1
        totalPlot = len(df.groupby(self.subplotVariableName))
        for outerkey, subDF in df.groupby(self.subplotVariableName):
            subplot = figure.add_subplot(1, totalPlot, numplot)
            for innerkey, lineDF in subDF.groupby(self.lineVariableName):
                plotDF = lineDF.reset_index()
                plotDF.plot(x=self.xVariableName, y=self.yVariableName,
                            ax=subplot, title="{}:{}".format(self.subplotVariableName, outerkey), label=innerkey)
            numplot += 1
        plt.xlabel(self.xVariableName)
        plt.ylabel(self.yVariableName)



def main():
    dataDir = "../data/evaluateNNvsLearningRateDecay"
    trainingDataDir = os.path.join(dataDir, "dataSets")
    dataSetName = "initPos=[30,30,20,20]_maxRunningSteps=30_numDataPoints=5800_numSimulations=200_numTrajs=200_rolloutSteps=10_standardizedReward=True.pickle"
    dataSetPath = os.path.join(trainingDataDir, dataSetName)
    if not os.path.exists(dataSetPath):
        print("No dataSet in:\n{}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)

    # NN fix Parameter
    fixedParameters = OrderedDict()
    fixedParameters['numStateSpace'] = 4
    fixedParameters['numActionSpace'] = 8
    fixedParameters['learningRate'] = 1e-4
    fixedParameters['regularizationFactor'] = 0
    fixedParameters['valueRelativeErrBound'] = 0.1
    fixedParameters['iteration'] = 10000
    fixedParameters['batchSize'] = 0
    fixedParameters['reportInterval'] = 1000
    fixedParameters['lossChangeThreshold'] = 1e-8
    fixedParameters['lossHistorySize'] = 10
    fixedParameters['initActionCoefficient'] = (1, 1)
    fixedParameters['initValueCoefficient'] = (1, 1)
    fixedParameters['neuronsPerLayer'] = 64
    fixedParameters['netLayers'] = 4

    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [100, 200, 300, 400]
    independentVariables['decayRate'] = [0.95]
    independentVariables['decayStep'] = [10, 100, 1000]

    # generate NN
    trainTerminalController = trainTools.TrainTerminalController(fixedParameters['lossHistorySize']
                                                                 , fixedParameters['lossChangeThreshold'])
    coefficientController = trainTools.coefficientCotroller(fixedParameters['initActionCoefficient']
                                                            , fixedParameters['initValueCoefficient'])
    trainReporter = trainTools.TrainReporter(fixedParameters['iteration']
                                             , fixedParameters['reportInterval'])
    generateLRModifier = lambda decayRate, decayStep: trainTools.learningRateModifier(fixedParameters['learningRate']
                                                                                      , decayRate, decayStep)
    generateTrain = lambda lrModifier: net.Train(fixedParameters['iteration'], fixedParameters['batchSize'], lrModifier
                                                 , trainTerminalController, coefficientController, trainReporter)
    generateModel = net.GenerateModelSeparateLastLayer(fixedParameters['numStateSpace'], fixedParameters['numActionSpace'], fixedParameters['regularizationFactor'],
                                                       valueRelativeErrBound=fixedParameters['valueRelativeErrBound'], seed=tfseed)
    model = generateModel([fixedParameters['neuronsPerLayer']] * fixedParameters['netLayers'])

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    extension = ".pickle"
    trainingOutputPath = os.path.join(dataDir, "trainingOutput")
    if not os.path.exists(trainingOutputPath):
        os.mkdir(trainingOutputPath)
    getSavePath = GetSavePath(trainingOutputPath, extension)
    generateTrainingOutput = GenerateTrainingLoss(getSavePath, model, generateTrain, generateLRModifier)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput, dataSet)
    generateDataFrameForDrawing = GenerateDataFrameForDrawing('trainingDataSize', 'actionLoss', 'decayRate', 'decayStep')
    dfs = generateDataFrameForDrawing(resultDF)
    plt.show()

if __name__ == "__main__":
    main()