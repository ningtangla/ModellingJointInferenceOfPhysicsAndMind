import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import pickle
from pylab import plt
import src.neuralNetwork.policyValueNet as net
from exec.evaluationFunctions import GetSavePath
import src.neuralNetwork.visualizeNN as visualizeNN
import src.neuralNetwork.trainTools as trainTools


class FetchParamsFromModel:
    def __init__(self, model, modelPath, feedDict, varNames, findKey):
        self.model = model
        self.modelPath = modelPath
        self.saver = model.graph.get_collection('saver')[0]
        self.feedDict = feedDict
        self.varNames = varNames
        self.findKey = findKey

    def __call__(self, oneConditionDf):
        iterNum = oneConditionDf.index.get_level_values('iterationNum')[0]
        self.saver.restore(self.model, f'{self.modelPath}-{iterNum}')
        sectionName, layerNum = oneConditionDf.index.get_level_values('layerIndex')[0]
        findKeyByVarName = lambda varName: self.findKey(varName, sectionName, layerNum)
        collectionKeys = [findKeyByVarName(varName) for varName in self.varNames]
        getTensorByKey = lambda key: self.model.graph.get_collection(key)[0]
        varNameToTensor = {varName: getTensorByKey(key) for varName, key in zip(self.varNames, collectionKeys)}
        varNameToValue = self.model.run(varNameToTensor, feed_dict=self.feedDict)
        varNameToValue = {name: value.flatten() for name, value in varNameToValue.items()}
        return pd.Series({'parameters': varNameToValue})


def main():
    # data
    dataSetDir = os.path.join('..', 'data', 'augmentDataForNN', 'dataSets')
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDir, dataSetExtension)
    pathVarDict = dict()
    pathVarDict['cBase'] = 100
    pathVarDict['numSimulations'] = 200
    pathVarDict["rolloutSteps"] = 10
    pathVarDict['initPos'] = 'Random'
    pathVarDict["numTrajs"] = 2500
    pathVarDict['maxRunningSteps'] = 30
    pathVarDict['numDataPoints'] = 68181
    pathVarDict["standardizedReward"] = True
    dataSetPath = getDataSetPath(pathVarDict)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)
    trainDataSize = 500
    trainData = [dataSet[varName][:trainDataSize] for varName in ['state', 'actionDist', 'value']]

    # model
    numStateSpace = 4
    numActionSpace = 8
    generateModel = net.GenerateModel(numStateSpace, numActionSpace)
    sharedWidths = [64, 64, 64]
    actionWidths = [64]
    valueWidths = [64]
    policyValueNet = generateModel(sharedWidths, actionWidths, valueWidths)
    sectionNameToDepth = {"shared": len(sharedWidths), "action": len(actionWidths) + 1, "value": len(valueWidths) + 1}
    layerIndices = visualizeNN.indexLayers(sectionNameToDepth)

    modelDir = os.path.join('..', 'data', 'iterationNumVSNNParameters', 'models', f'{trainDataSize}DataSize')
    modelExtension = ''
    getModelPath = GetSavePath(modelDir, modelExtension)
    modelVarDict = dict()
    modelVarDict['structure'] = (tuple(sharedWidths), tuple(actionWidths), tuple(valueWidths))
    modelVarDict['dataSize'] = trainDataSize
    modelPath = getModelPath(modelVarDict)
    saver = policyValueNet.graph.get_collection("saver")[0]
    saver.save(policyValueNet, modelPath, global_step=0)
    pretrainedIterationNum = 100000
    saver.restore(policyValueNet, f'{modelPath}-{pretrainedIterationNum}')

    # train
    lossHistorySize = 50
    lossChangeThreshold = 1e-8
    trainTerminalController = trainTools.TrainTerminalController(lossHistorySize, lossChangeThreshold)

    initCoeffs = (1, 1)
    coefficientController = trainTools.CoefficientCotroller(initCoeffs, initCoeffs)

    learningRate = 1e-4
    decayRate = 1
    decayStep = 10000
    learningRateModifier = trainTools.LearningRateModifier(learningRate, decayRate, decayStep)

    maxStepNum = 2000
    reportInterval = 1000
    trainReporter = trainTools.TrainReporter(maxStepNum, reportInterval)

    batchSize = 0
    sampleData = net.sampleData
    train = net.Train(maxStepNum, batchSize, sampleData, learningRateModifier, trainTerminalController,
                      coefficientController, trainReporter)

    rep = 50
    ckptSteps = list(range(maxStepNum + pretrainedIterationNum,
                           rep * maxStepNum + pretrainedIterationNum + 1, maxStepNum))
    saveModelsOn = False
    if saveModelsOn:
        for step in ckptSteps:
            trainedModel = train(policyValueNet, trainData)
            saver = trainedModel.graph.get_collection_ref("saver")[0]
            savePath = saver.save(trainedModel, modelPath, global_step=step, write_meta_graph=False)
            print(f"Model saved in {savePath}")
        exit()

    # empty data frame
    manipulatedVariables = OrderedDict()
    manipulatedVariables['iterationNum'] = list(range(0, 20001, 2000))
    manipulatedVariables['layerIndex'] = layerIndices
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # function to apply
    graph = policyValueNet.graph
    states = graph.get_collection("inputs")[0]
    groundTruthActions, groundTruthValues = graph.get_collection("groundTruths")
    feedDict = {states: trainData[0], groundTruthActions: trainData[1], groundTruthValues: trainData[2]}
    varNames = ['weight', 'bias', 'activation', 'weightGradient', 'biasGradient']
    allKeys = graph.get_all_collection_keys()
    findKey = visualizeNN.FindKey(allKeys)
    fetchTensorValue = FetchParamsFromModel(policyValueNet, modelPath, feedDict, varNames, findKey)

    nnVariableDf = toSplitFrame.groupby(levelNames).apply(fetchTensorValue)
    resultDfDir = os.path.join('..', 'data', 'iterationNumVSNNParameters', 'results')
    saveDf = False
    if saveDf:
        with open(os.path.join(resultDfDir, f'dataSize={trainDataSize}.pkl'), 'wb') as f:
            pickle.dump(nnVariableDf, f)
    print(nnVariableDf)
    exit()

    # plot
    useAbs = True
    useLog = False
    histBase = 1e-10
    binCount = 50
    varNameToPlot = 'weightGradient'
    rawAllData = np.concatenate(nnVariableDf[varNameToPlot].tolist())
    allData = np.abs(rawAllData) if useAbs else rawAllData
    _, bins = visualizeNN.logHist(allData, binCount, histBase) if useLog else np.histogram(allData, bins=binCount)

    plotBars = visualizeNN.PlotBars(useAbs, useLog)

    # bar
    barFig = plt.figure()
    barFig.suptitle(f"Bar plots of {varNameToPlot}, data size: {trainDataSize}, nn structure: {modelVarDict['structure']}")
    figRowNum = 1
    figColNum = len(manipulatedVariables['iterationNum'])
    subplotIndex = 1
    axs = []
    for iterNum, iterGrp in nnVariableDf.groupby('iterationNum'):
        iterGrp.index = iterGrp.index.droplevel('iterationNum')
        rawDataList = [nnVariableDf[varNameToPlot][iterNum][layerIndex] for layerIndex in layerIndices]
        dataList = [np.abs(data) for data in rawDataList] if useAbs else rawDataList
        indexToStr = lambda sectionName, layerIndex: f"{sectionName}/{layerIndex}"
        labelList = [indexToStr(section, layer) for section, layer in layerIndices]
        statsOnPlot = [(np.mean(data), np.std(data), np.min(data), np.max(data)) for data in dataList]
        means, stds, mins, maxs = [np.array(stats) for stats in zip(*statsOnPlot)]
        ax = barFig.add_subplot(figRowNum, figColNum, subplotIndex)
        axs.append(ax)
        subplotIndex += 1
        ax.set_title(f'# of iterations = {iterNum}')
        plotBars(ax, means, stds, mins, maxs, labelList)
    visualizeNN.syncLimits(axs)
    plt.show()


if __name__ == "__main__":
    main()
