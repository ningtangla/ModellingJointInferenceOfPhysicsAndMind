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
        initCoeffs = df.index.get_level_values('initCoeffs')[0]
        netNeurons = df.index.get_level_values('netNeurons')[0]
        netLayers = df.index.get_level_values('netLayers')[0]
        neuronsPerLayer = int(round(netNeurons/netLayers))

        generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
        model = generateModel([neuronsPerLayer] * netLayers)

        trainDataValueType = "standardized" if useStandardizedReward else "unstandardized"
        modelName = "{}data_{}x{}_{}kIter_{}Value_initCoefs={}".format(len(trainData[0]), neuronsPerLayer, netLayers, round(maxStepNum / 1000), trainDataValueType, initCoeffs)
        modelName = modelName.replace(' ', '')
        modelPath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
        trainedModel = net.restoreVariables(model, modelPath)

        evalTrain = net.evaluate(trainedModel, trainData)
        evalTest = net.evaluate(trainedModel, testData)
        return pd.Series({"testActionLoss": evalTest['actionLoss'], "trainActionLoss": evalTrain["actionLoss"],
                          "testActionAcc": evalTest["actionAcc"], "trainActionAcc": evalTrain["actionAcc"],
                          "testValueLoss": evalTest["valueLoss"], "trainValueLoss": evalTrain["valueLoss"],
                          "testValueAcc": evalTest["valueAcc"], "trainValueAcc": evalTrain["valueAcc"]})


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
    independentVariables['trainingDataSize'] = [3000]
    independentVariables['testDataSize'] = [2800]
    independentVariables['numStateSpace'] = [4]
    independentVariables['numActionSpace'] = [8]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['regularizationFactor'] = [0]
    independentVariables['valueRelativeErrBound'] = [0.1]
    independentVariables['iteration'] = [50000]
    independentVariables['initCoeffs'] = [(1, 1), (10, 1), (50, 1)]
    independentVariables['netNeurons'] = [256]
    independentVariables['netLayers'] = [4]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    saveModelDir = "../../data/compareValueDataStandardizationAndLossCoefs/trainedModels"
    applyFunctoin = ApplyFunction(saveModelDir)
    evalResults = toSplitFrame.groupby(levelNames).apply(applyFunctoin, getDataSetPath, pathVarDict, None, tfseed)

    numCondtions = 3
    for i in range(numCondtions):
        print(evalResults.iloc[i])

    evalResultsDir = "../../data/compareValueDataStandardizationAndLossCoefs/evalWithDataSetResults"
    if not os.path.exists(evalResultsDir):
        os.makedirs(evalResultsDir)
    evalResultsFileName = "diffLossCoefs.pickle"
    evalResultsPath = os.path.join(evalResultsDir, evalResultsFileName)
    with open(evalResultsPath, 'wb') as f:
        pickle.dump(evalResults, f)
    print("Evaluation results saved in {}".format(evalResultsPath))

    stats1 = [evalResults.iloc[0][i] for i in range(1, 8, 2)]
    stats2 = [evalResults.iloc[1][i] for i in range(1, 8, 2)]
    stats3 = [evalResults.iloc[2][i] for i in range(1, 8, 2)]

    barWidth = 0.25
    r1 = np.arange(len(stats1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, stats1, width=barWidth, label='(1, 1)')
    plt.bar(r2, stats2, width=barWidth, label='(10, 1)')
    plt.bar(r3, stats3, width=barWidth, label='(50, 1)')
    plt.xlabel('measures on training data')
    plt.xticks([r + barWidth / 2.0 for r in range(len(stats1))], ['actionLoss', 'actionAcc', 'valueLoss', 'valueAcc'])
    title = 'NN trained with different loss coefficients'
    plt.title(title)
    plt.legend(title="loss coefficients")
    figPath = os.path.join(evalResultsDir, title + ".png")
    plt.savefig(figPath)


if __name__ == "__main__":
    main()
