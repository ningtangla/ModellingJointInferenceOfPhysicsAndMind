import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd
import pickle
from pylab import plt
import src.neuralNetwork.policyValueNet as net
from exec.evaluationFunctions import GetSavePath


def frameNNData(levelNames, sectionNameToDepth):
    indexSection = lambda name, numOfLayers: [(name, i + 1) for i in range(numOfLayers)]
    sectionIndexLists = [indexSection(sectionName, depth) for sectionName, depth in sectionNameToDepth.items()]
    layerIndices = sum(sectionIndexLists, [])
    multiIndex = pd.MultiIndex.from_tuples(layerIndices, names=levelNames)
    toSplitFrame = pd.DataFrame(index=multiIndex)
    return toSplitFrame


class FindKey:
    def __init__(self, allKeys):
        self.allKeys = allKeys

    def __call__(self, varName, sectionName, layerIndex):
        keyPrefix = f"{varName}/{sectionName}/fc{layerIndex}/"
        matches = np.array([keyPrefix in key for key in self.allKeys])
        matchIndices = np.argwhere(matches).flatten()
        assert(len(matchIndices) == 1)
        matchKey = self.allKeys[matchIndices[0]]
        return matchKey


class FetchVarsFromModel:
    def __init__(self, model, feedDict, varNames, findKey):
        self.model = model
        self.feedDict = feedDict
        self.varNames = varNames
        self.findKey = findKey

    def __call__(self, oneConditionDf):
        sectionName = oneConditionDf.index.get_level_values("section")[0]
        layerIndex = oneConditionDf.index.get_level_values("layer")[0]
        findKeyByVarName = lambda varName: self.findKey(varName, sectionName, layerIndex)
        collectionKeys = [findKeyByVarName(varName) for varName in self.varNames]
        getTensorByKey = lambda key: self.model.graph.get_collection(key)[0]
        varNameToTensor = {varName: getTensorByKey(key) for varName, key in zip(self.varNames, collectionKeys)}
        varNameToValue = self.model.run(varNameToTensor, feed_dict=self.feedDict)
        varNameToValue = {name: value.flatten() for name, value in varNameToValue.items()}
        return pd.Series(varNameToValue)


def logHist(data, bins, base):
    logData = np.log10(np.array(data) + base)
    counts, logBins = np.histogram(logData, bins=bins)
    return counts, 10 ** logBins


def syncLimits(axs):
    xlims = sum([list(ax.get_xlim()) for ax in axs], [])
    xlim = (min(xlims), max(xlims))
    ylims = sum([list(ax.get_ylim()) for ax in axs], [])
    ylim = (min(ylims), max(ylims))
    newLimits = [(ax.set_xlim(xlim), ax.set_ylim(ylim)) for ax in axs]
    return newLimits


class PlotHist:
    def __init__(self, useAbs, useLog, binCount, histBase, syncLimits):
        self.useAbs = useAbs
        self.useLog = useLog
        self.binCount = binCount
        self.histBase = histBase
        self.syncLimits = syncLimits

    def __call__(self, nnVariableDf, varName, sectionNameToDepth):
        plotRows = len(sectionNameToDepth)
        plotCols = max(list(sectionNameToDepth.values()))
        fig = plt.figure()
        fig.suptitle(f"Histograms of {varName}")
        gs = fig.add_gridspec(plotRows, plotCols)

        rawAllData = np.concatenate(nnVariableDf[varName].tolist())
        allData = np.abs(rawAllData) if self.useAbs else rawAllData
        counts, bins = logHist(allData, self.binCount, self.histBase) if self.useLog \
            else np.histogram(allData, bins=self.binCount)

        sectionNameToRow = {'shared': 0, 'action': 1, 'value': 2}
        axs = []
        for sectionName, sectionGrp in nnVariableDf.groupby('section'):
            sectionGrp.index = sectionGrp.index.droplevel('section')
            for layerIndex, layerDf in sectionGrp.groupby('layer'):
                axForDraw = fig.add_subplot(gs[sectionNameToRow[sectionName], layerIndex - 1])
                if self.useLog:
                    axForDraw.set_xscale("log")
                axs.append(axForDraw)
                rawData = np.array(layerDf[varName][layerIndex])
                data = (np.abs(rawData) if self.useAbs else rawData) + (self.histBase if self.useLog else 0)
                counts, _ = np.histogram(data, bins=bins)
                axForDraw.hist(bins[:-1], bins=bins, weights=counts / np.sum(counts))
                axForDraw.set_title(f"{sectionName}/{layerIndex} $\mu=${np.mean(data):.2E} $\sigma=${np.std(data):.2E}")
        self.syncLimits(axs)
        plt.show()


class PlotErrorBars:
    def __init__(self, useAbs, useLog):
        self.useAbs = useAbs
        self.useLog = useLog

    def __call__(self, nnVariableDf, varName, sectionNameToDepth):
        indexSection = lambda name, numOfLayers: [(name, i + 1) for i in range(numOfLayers)]
        sectionIndexLists = [indexSection(sectionName, depth) for sectionName, depth in sectionNameToDepth.items()]
        layerIndices = sum(sectionIndexLists, [])
        numLayers = len(layerIndices)

        rawDataList = [nnVariableDf[varName][section][layer] for section, layer in layerIndices]
        dataList = [np.abs(data) for data in rawDataList] if self.useAbs else rawDataList
        indexToStr = lambda sectionName, layerIndex: f"{sectionName}/{layerIndex}"
        labelList = [indexToStr(section, layer) for section, layer in layerIndices]

        fig, ax = plt.subplots()
        if self.useLog:
            ax.set_yscale('log')
        statsOnPlot = [(np.mean(data), np.std(data), np.min(data), np.max(data)) for data in dataList]
        means, stds, mins, maxs = [np.array(stats) for stats in zip(*statsOnPlot)]
        ax.plot(range(numLayers), means, 'or', label="$\mu$")
        ax.errorbar(range(numLayers), means, stds, label="$\sigma$", fmt='.', markersize=0, ecolor="black", lw=4)
        ax.errorbar(range(numLayers), means, [means - mins, maxs - means], label="range", fmt=".", markersize=0, ecolor="grey", lw=1.5)
        ax.legend()
        ax.set_xticks(range(numLayers))
        ax.set_xticklabels(labelList)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.set_title("Bar plots of {}".format(varName))
        plt.show()


def main():
    # model
    numStateSpace = 4
    numActionSpace = 8
    generateModel = net.GenerateModel(numStateSpace, numActionSpace)
    sharedWidths = [100]
    actionWidths = [300, 300, 200]
    valueWidths = [200]
    policyValueNet = generateModel(sharedWidths, actionWidths, valueWidths)

    # empty data frame to split
    levelNames = ["section", "layer"]
    sectionNameToDepth = {"shared": len(sharedWidths), "action": len(actionWidths) + 1, "value": len(valueWidths) + 1}
    toSplitFrame = frameNNData(levelNames, sectionNameToDepth)

    # data
    dataSetsDir = '../../data/compareValueDataStandardizationAndLossCoefs/trainingData/dataSets'
    extension = '.pickle'
    getDataSetPath = GetSavePath(dataSetsDir, extension)
    initPosition = np.array([[30, 30], [20, 20]])
    maxRollOutSteps = 10
    numSimulations = 200
    maxRunningSteps = 30
    numTrajs = 200
    numDataPoints = 5800
    pathVarDict = dict()
    pathVarDict["initPos"] = list(initPosition.flatten())
    pathVarDict["rolloutSteps"] = maxRollOutSteps
    pathVarDict["numSimulations"] = numSimulations
    pathVarDict["maxRunningSteps"] = maxRunningSteps
    pathVarDict["numTrajs"] = numTrajs
    pathVarDict["numDataPoints"] = numDataPoints
    pathVarDict["standardizedReward"] = True
    savePath = getDataSetPath(pathVarDict)
    with open(savePath, "rb") as f:
        dataSet = pickle.load(f)
    trainDataSize = 3000
    trainData = [dataSet[varName][:trainDataSize] for varName in ['state', 'actionDist', 'value']]

    # function to apply
    graph = policyValueNet.graph
    states = graph.get_collection("inputs")[0]
    groundTruthActions, groundTruthValues = graph.get_collection("groundTruths")
    feedDict = {states: trainData[0], groundTruthActions: trainData[1], groundTruthValues: trainData[2]}
    varNames = ['weight', 'bias', 'activation', 'weightGradient', 'biasGradient']
    allKeys = graph.get_all_collection_keys()
    findKey = FindKey(allKeys)
    fetchTensorValue = FetchVarsFromModel(policyValueNet, feedDict, varNames, findKey)

    nnVariableDf = toSplitFrame.groupby(levelNames).apply(fetchTensorValue)
    print(nnVariableDf)

    # plot
    useAbs = True
    useLogScale = True
    binCount = 50
    histBase = 1e-10
    plotHist = PlotHist(useAbs, useLogScale, binCount, histBase, syncLimits)
    plotBars = PlotErrorBars(useAbs, useLogScale)

    varNameToPlot = 'weightGradient'
    plotHist(nnVariableDf, varNameToPlot, sectionNameToDepth)


if __name__ == "__main__":
    main()
