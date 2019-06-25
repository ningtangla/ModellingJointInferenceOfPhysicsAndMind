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
import matplotlib.pyplot as plt
import policyValueNet as net
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
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainDataActionType = df.index.get_level_values('trainingDataActionType')[0]
        trainData = [dataSet[varName][:trainDataSize] for varName in ['state', trainDataActionType, 'value']]
        testDataSize = df.index.get_level_values('testDataSize')[0]
        testData = [dataSet[varName][:testDataSize] for varName in ['state', trainDataActionType, 'value']]
        numStateSpace = df.index.get_level_values('numStateSpace')[0]
        numActionSpace = df.index.get_level_values('numActionSpace')[0]
        learningRate = df.index.get_level_values('learningRate')[0]
        regularizationFactor = df.index.get_level_values('regularizationFactor')[0]
        valueRelativeErrBound = df.index.get_level_values('valueRelativeErrBound')[0]
        maxStepNum = df.index.get_level_values('iteration')[0]
        netNeurons = df.index.get_level_values('netNeurons')[0]
        netLayers = df.index.get_level_values('netLayers')[0]
        neuronsPerLayer = int(round(netNeurons/netLayers))

        generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
        model = generateModel([neuronsPerLayer] * netLayers)

        trainDataValueType = "standardized" if useStandardizedReward else "unstandardized"
        modelName = "{}data_{}x{}_{}kIter_{}Value".format(len(trainData[0]), neuronsPerLayer, netLayers, round(maxStepNum / 1000), trainDataValueType)
        modelPath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
        trainedModel = net.restoreVariables(model, modelPath)

        evalTrain = net.evaluate(trainedModel, trainData)
        evalTest = net.evaluate(trainedModel, testData)
        return pd.Series({"testActionLoss": evalTest['actionLoss'], "trainActionLoss": evalTrain["actionLoss"],
                          "testActionAcc": evalTest["actionAcc"], "trainActionAcc": evalTrain["actionAcc"],
                          "testValueLoss": evalTest["valueLoss"], "trainValueLoss": evalTrain["valueLoss"],
                          "testValueAcc": evalTest["valueAcc"], "trainValueAcc": evalTrain["valueAcc"]})


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
    independentVariables['testDataSize'] = [2800]
    independentVariables['numStateSpace'] = [4]
    independentVariables['numActionSpace'] = [8]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['regularizationFactor'] = [0]
    independentVariables['valueRelativeErrBound'] = [0.1]
    independentVariables['iteration'] = [50000]
    independentVariables['batchSize'] = [3000]
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
    evalResults = toSplitFrame.groupby(levelNames).apply(applyFunctoin, getDataSetPath, pathVarDict, None, tfseed)

    evalResultsDir = "../data/evaluateNNWithDataSet_standardizedValueDataVSunstandardizedValueData"
    if not os.path.exists(evalResultsDir):
        os.makedirs(evalResultsDir)
    evalResultsFileName = "evalResults.pickle"
    evalResultsPath = os.path.join(evalResultsDir, evalResultsFileName)
    with open(evalResultsPath, 'wb') as f:
        pickle.dump(evalResults, f)

    unstandardizedStats = [evalResults.iloc[0][i] for i in range(1, 8, 2)]
    standardizedStats = [evalResults.iloc[1][i] for i in range(1, 8, 2)]

    barWidth = 0.25
    r1 = np.arange(len(unstandardizedStats))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, unstandardizedStats, width=barWidth, label='unstandardized reward')
    plt.bar(r2, standardizedStats, width=barWidth, label='standardized reward')
    plt.xlabel('measures on training data')
    plt.xticks([r + barWidth / 2.0 for r in range(len(unstandardizedStats))], ['actionLoss', 'actionAcc', 'valueLoss', 'valueAcc'])
    title = 'NN trained with unstandardized value data vs standardized value data'
    plt.title(title)
    plt.legend()
    figPath = os.path.join(evalResultsDir, title + ".png")
    plt.savefig(figPath)


if __name__ == "__main__":
    main()
