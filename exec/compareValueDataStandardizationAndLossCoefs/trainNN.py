import sys
sys.path.append("..")
sys.path.append("../../src/neuralNetwork")
sys.path.append("../../src/constrainedChasingEscapingEnv")
sys.path.append("../../src/algorithms")
sys.path.append("../../src")
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import pickle
import flexiblePolicyValueNet as net
import trainTools
from evaluationFunctions import GetSavePath


class ApplyFunction:
    def __init__(self, saveModelDir=None):
        self.saveModelDir = saveModelDir

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
        initCoeffs = df.index.get_level_values('initCoeffs')[0]
        netNeurons = df.index.get_level_values('netNeurons')[0]
        netLayers = df.index.get_level_values('netLayers')[0]
        neuronsPerLayer = int(round(netNeurons/netLayers))
        reportInterval = df.index.get_level_values('reportInterval')[0]

        sampleData = net.sampleData
        trainTerminalController = trainTools.TrainTerminalController(lossHistorySize, lossChangeThreshold)
        coefficientController = trainTools.CoefficientCotroller(initCoeffs, initCoeffs)
        decayRate = 1
        decayStep = 10000
        learningRateModifier = trainTools.LearningRateModifier(learningRate, decayRate, decayStep)
        trainReporter = trainTools.TrainReporter(maxStepNum, reportInterval)
        train = net.Train(maxStepNum, batchSize, sampleData, learningRateModifier, trainTerminalController, coefficientController, trainReporter)

        generateModel = net.GenerateModel(numStateSpace, numActionSpace, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
        model = generateModel([neuronsPerLayer] * (netLayers-1), [neuronsPerLayer], [neuronsPerLayer])
        trainedModel = train(model, trainData)
        trainDataValueType = "standardized" if useStandardizedReward else "unstandardized"
        modelName = "newNet_{}data_{}x{}_{}kIter_{}Value_initCoefs={}".format(len(trainData[0]), neuronsPerLayer, netLayers, round(maxStepNum / 1000), trainDataValueType, initCoeffs)
        modelName = modelName.replace(' ', '')
        if self.saveModelDir is not None:
            savePath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
            net.saveVariables(trainedModel, savePath)


def main(tfseed=128):
    dataSetsDir = '../../data/compareValueDataStandardizationAndLossCoefs/trainingData/dataSets'
    extension = ".pickle"
    getDataSetPath = GetSavePath(dataSetsDir, extension)

    initPosition = np.array([[30, 30], [20, 20]])
    maxRollOutSteps = 10
    numSimulations = 200
    maxRunningSteps = 30
    numTrajs = 200
    numDataPoints = 5800
    cBase = 100
    pathVarDict = {}
    pathVarDict["initPos"] = list(initPosition.flatten())
    pathVarDict["rolloutSteps"] = maxRollOutSteps
    pathVarDict["numSimulations"] = numSimulations
    pathVarDict["maxRunningSteps"] = maxRunningSteps
    pathVarDict["numTrajs"] = numTrajs
    pathVarDict["numDataPoints"] = numDataPoints
    # pathVarDict["cBase"] = cBase

    independentVariables = OrderedDict()
    independentVariables['useStandardizedReward'] = [True]
    independentVariables['trainingDataActionType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [1500]
    independentVariables['numStateSpace'] = [4]
    independentVariables['numActionSpace'] = [8]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['regularizationFactor'] = [0]
    independentVariables['valueRelativeErrBound'] = [0.1]
    independentVariables['iteration'] = [1000]
    independentVariables['batchSize'] = [0]
    independentVariables['reportInterval'] = [1000]
    independentVariables['lossChangeThreshold'] = [0]
    independentVariables['lossHistorySize'] = [10]
    independentVariables['initCoeffs'] = [(1, 1)]
    independentVariables['netNeurons'] = [256]
    independentVariables['netLayers'] = [4]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    saveModelDir = "../../data/compareValueDataStandardizationAndLossCoefs/trainedModels"
    applyFunctoin = ApplyFunction(saveModelDir)
    toSplitFrame.groupby(levelNames).apply(applyFunctoin, getDataSetPath, pathVarDict, None, tfseed)


if __name__ == "__main__":
    main()
