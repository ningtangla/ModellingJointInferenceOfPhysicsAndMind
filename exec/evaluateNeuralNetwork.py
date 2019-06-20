import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/sheepWolf")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import pandas as pd
import random
import os
import pickle
import policyValueNet as net
import trainTools
from collections import OrderedDict


class ApplyFunction:
    def __init__(self, saveModelDir=None, saveGraphDir=None):
        self.saveModelDir = saveModelDir
        self.saveGraphDir = saveGraphDir

    def __call__(self, df, dataSet, criticFunction, tfseed):
        trainDataSize = df.index.get_level_values('trainingDataSize')[0]
        trainData =  [list(varData) for varData in zip(*dataSet[:trainDataSize])]
        testDataSize = df.index.get_level_values('testDataSize')[0]
        testData = [list(varData) for varData in zip(*dataSet[:testDataSize])]
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
        modelName = "{}data_{}x{}_minibatch_{}kIter_contState_actionDist".format(len(trainData[0]), neuronsPerLayer, netLayers, round(maxStepNum / 1000))
        if self.saveModelDir is not None:
            savePath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
            net.saveVariables(trainedModel, savePath)

        evalTest = net.evaluate(trainedModel, testData)
        return pd.Series({"testActionLoss": evalTest['actionLoss']})


def main(tfseed=128):
    saveDir = "../data"
    saveModelDir = "../data/neuralNetworkGraphVariables"
    dataSetName = "test"
    dataSetPath = os.path.join(saveDir, dataSetName)
    dataSet =
    random.shuffle(dataSet)

    independentVariables = OrderedDict()
    independentVariables['trainingDataSize'] = [len(dataSet)]  # [5000, 15000, 30000, 45000, 60000]
    independentVariables['testDataSize'] = [10000]
    independentVariables['numStateSpace'] = [4]
    independentVariables['numActionSpace'] = [8]
    independentVariables['learningRate'] = [1e-4]
    independentVariables['regularizationFactor'] = [0]
    independentVariables['valueRelativeErrBound'] = [0.1]
    independentVariables['iteration'] = [50000]
    independentVariables['batchSize'] = [4096]
    independentVariables['reportInterval'] = [1000]
    independentVariables['lossChangeThreshold']= [1e-8]
    independentVariables['lossHistorySize'] = [10]
    independentVariables['initActionCoefficient'] = [50]
    independentVariables['initValueCoefficient'] = [1]
    independentVariables['netNeurons'] = [256]
    independentVariables['netLayers'] = [4]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    applyFunctoin = ApplyFunction(saveModelDir)
    resultDF = toSplitFrame.groupby(levelNames).apply(applyFunctoin, dataSet, None, tfseed)
    file = open("temp.pkl", "wb")
    pickle.dump(resultDF, file)


if __name__ == "__main__":
    main()
