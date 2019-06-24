import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import pickle
import policyValueNet as net
import trainTools
from evaluationFunctions import GetSavePath


class ApplyFunction:
    def __init__(self, saveModelDir=None, saveGraphDir=None):
        self.saveModelDir = saveModelDir
        self.saveGraphDir = saveGraphDir

    def __call__(self, df, getDataSetPath, pathVarDict, criticFunction, tfseed):
        useStandardizedReward = df.index.get_level_values('useStandardizedReward')[0]
        pathVarDict["standardizedReward"] = useStandardizedReward
        dataSetPath = getDataSetPath(pathVarDict)
        with open(dataSetPath, "rb") as f:
            dataSet = pickle.load(f)
        print("Loaded data set from {}".format(dataSetPath))
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataActionType = df.index.get_level_values('trainingDataActionType')[0]
        trainData = [dataSet[varName][:trainDataSize] for varName in ['state', trainDataActionType, 'value']]
        numStateSpace = df.index.get_level_values('numStateSpace')[0]
        numActionSpace = df.index.get_level_values('numActionSpace')[0]
        learningRate = df.index.get_level_values('learningRate')[0]
        regularizationFactor = df.index.get_level_values('regularizationFactor')[0]
        valueRelativeErrBound = df.index.get_level_values('valueRelativeErrBound')[0]
        maxStepNum = df.index.get_level_values('iteration')[0]
        batchSize = df.index.get_level_values('batchSize')[0]
        lossChangeThreshold = df.index.get_level_values('lossChangeThreshold')[0]
        lossHistorySize = df.index.get_level_values('lossHistorySize')[0]
        initActionCoefficient = df.index.get_level_values('initActionCoefficient')[0]
        initValueCoefficient = df.index.get_level_values('initValueCoefficient')[0]
        netNeurons = df.index.get_level_values('netNeurons')[0]
        netLayers = df.index.get_level_values('netLayers')[0]
        neuronsPerLayer = int(round(netNeurons/netLayers))
        reportInterval = df.index.get_level_values('reportInterval')[0]

        trainTerminalController = trainTools.TrainTerminalController(lossHistorySize, lossChangeThreshold)
        coefficientController = trainTools.coefficientCotroller(initActionCoefficient, initValueCoefficient)
        trainReporter = trainTools.TrainReporter(maxStepNum, reportInterval)
        train = net.Train(maxStepNum, batchSize, trainTerminalController, coefficientController, trainReporter)

        generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
        model = generateModel([neuronsPerLayer] * netLayers)
        trainedModel = train(model, trainData)
        trainDataValueType = "standardized" if useStandardizedReward else "unstandardized"
        modelName = "{}data_{}x{}_{}kIter_{}Value".format(len(trainData[0]), neuronsPerLayer, netLayers, round(maxStepNum / 1000), trainDataValueType)
        if self.saveModelDir is not None:
            savePath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
            net.saveVariables(trainedModel, savePath)


def main(tfseed=128):
    dataSetsDir = '../data/trainingDataForNN/dataSets'
    extension = ".pickle"
    getDataSetPath = GetSavePath(dataSetsDir, extension)

    initPosition = np.array([[30, 30], [20, 20]])
    maxRollOutSteps = 10
    numSimulations = 200
    maxRunningSteps = 30
    numTrajs = 200
    numDataPoints = 5800
    pathVarDict = {}
    pathVarDict["initPos"] = list(initPosition.flatten())
    pathVarDict["rolloutSteps"] = maxRollOutSteps
    pathVarDict["numSimulations"] = numSimulations
    pathVarDict["maxRunningSteps"] = maxRunningSteps
    pathVarDict["numTrajs"] = numTrajs
    pathVarDict["numDataPoints"] = numDataPoints

    independentVariables = OrderedDict()
    independentVariables['useStandardizedReward'] = [True, False]
    independentVariables['trainingDataActionType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [3000]  # [5000, 15000, 30000, 45000, 60000]
    independentVariables['numStateSpace'] = [4]
    independentVariables['numActionSpace'] = [8]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['regularizationFactor'] = [0]
    independentVariables['valueRelativeErrBound'] = [0.1]
    independentVariables['iteration'] = [50000]
    independentVariables['batchSize'] = [0]
    independentVariables['reportInterval'] = [1000]
    independentVariables['lossChangeThreshold'] = [1e-8]
    independentVariables['lossHistorySize'] = [10]
    independentVariables['initActionCoefficient'] = [1]
    independentVariables['initValueCoefficient'] = [1]
    independentVariables['netNeurons'] = [256]
    independentVariables['netLayers'] = [4]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    saveModelDir = "../data/neuralNetworkGraphVariables"
    applyFunctoin = ApplyFunction(saveModelDir)
    toSplitFrame.groupby(levelNames).apply(applyFunctoin, getDataSetPath, pathVarDict, None, tfseed)


if __name__ == "__main__":
    main()
