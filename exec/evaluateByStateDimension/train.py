import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import pandas as pd
import policyValueNet as net
import pickle
from collections import OrderedDict
from trajectoriesSaveLoad import GetSavePath
import trainTools
import random

def dictToFileName(parameters):
    sortedParameters = sorted(parameters.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1])
        for parameter in sortedParameters
    ]
    modelName = '_'.join(nameValueStringPairs).replace(" ", "")
    return modelName


class GenerateTrainedModel:

    def __init__(self, getSavePathForModel, getSavePathForDataSet, getGenerateModel, generateTrain):
        self.getSavePathForModel = getSavePathForModel
        self.generateTrain = generateTrain
        self.getGenerateModel = getGenerateModel
        self.getSavePathForDataSet = getSavePathForDataSet

    def __call__(self, df):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        batchSize = df.index.get_level_values('batchSize')[0]
        trainingStep = df.index.get_level_values('trainingStep')[0]
        neuronsPerLayer = df.index.get_level_values('neuronsPerLayer')[0]
        sharedLayers = df.index.get_level_values('sharedLayers')[0]
        actionLayers = df.index.get_level_values('actionLayers')[0]
        valueLayers = df.index.get_level_values('valueLayers')[0]
        numOfFrame = df.index.get_level_values('numOfFrame')[0]
        numOfStateSpace = df.index.get_level_values('numOfStateSpace')[0]
        trainingDataType = df.index.get_level_values('trainingDataType')[0]
        generateModel = self.getGenerateModel(numOfStateSpace*numOfFrame)
        dataSetparameters = {"numOfStateSpace":numOfStateSpace,
                             "numOfFrame":numOfFrame,
                             "trainingDataType":trainingDataType}
        dataSetPath = self.getSavePathForDataSet(dataSetparameters)
        with open(dataSetPath, 'rb') as f:
            dataSet = pickle.load(f)
        trainData = [element[:trainDataSize] for element in dataSet]
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        saveModelDir = self.getSavePathForModel(parameters)
        modelName = dictToFileName(parameters)
        modelPath = os.path.join(saveModelDir, modelName)
        model = generateModel([neuronsPerLayer] * sharedLayers,
                            [neuronsPerLayer] * actionLayers,
                            [neuronsPerLayer] * valueLayers)
        if not os.path.exists(saveModelDir):
            train = self.generateTrain(trainingStep, batchSize)
            trainedModel = train(model, trainData)
            net.saveVariables(trainedModel, modelPath)
        return pd.Series({"train": 'done'})



def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'evaluateByStateDimension')

    # generate NN model
    numOfActionSpace = 8
    regularizationFactor = 1e-4
    getGenerateModel = lambda numOfStateSpace: net.GenerateModel(numOfStateSpace, numOfActionSpace, regularizationFactor)

    # generate NN train
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    trainTerminalController = trainTools.TrainTerminalController(
        lossHistorySize, lossChangeThreshold)
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
    generateTrain = lambda trainingStep, batchSize: net.Train(
        trainingStep, batchSize, net.sampleData, learningRateModifier,
        trainTerminalController, coefficientController, reporter)

    # split
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['trainingDataSize'] = [num for num in range(2000, 20001, 2000)]
    independentVariables['batchSize'] = [64, 256]
    independentVariables['augment'] = [False]
    independentVariables['trainingStep'] = [10000]
    independentVariables['neuronsPerLayer'] = [128]
    independentVariables['sharedLayers'] = [1]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]
    independentVariables['numOfFrame'] = [1, 3]
    independentVariables['numOfStateSpace'] = [8]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    trainedModelDir = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)
    getModelSavePath= GetSavePath(trainedModelDir, "")

    # get train data
    trainingDataDir = os.path.join(dataDir, 'trainingData')
    getDataSavePath = GetSavePath(trainingDataDir, ".pickle")

    generateTrainingOutput = GenerateTrainedModel(getModelSavePath,
                                                  getDataSavePath,
                                                  getGenerateModel,
                                                  generateTrain)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateTrainingOutput)


if __name__ == "__main__":
    main()
