import sys
sys.path.append('../..')
import numpy as np
import pickle
from pylab import plt
import src.neuralNetwork.policyValueNet as net
from exec.evaluationFunctions import GetSavePath


def summarizeStats(data):
    data = np.array(data).flatten()
    summary = dict()
    summary["mean"] = np.mean(data)
    summary["std"] = np.std(data)
    summary["min"] = np.min(data)
    summary["med"] = np.median(data)
    summary["max"] = np.max(data)
    return summary


def clipTensorName(name):
    nameComponents = name.split('/')[:-1]
    clippedName = '/'.join(nameComponents)
    return clippedName


class FetchTensorValuesInOneCollection:
    def __init__(self, model, rename):
        self.model = model
        self.rename = rename

    def __call__(self, collectionName, feedDict=None):
        g = self.model.graph
        tensors_ = g.get_collection(collectionName)
        nameToTensor_ = {self.rename(t_.name): t_ for t_ in tensors_}
        nameToValue = self.model.run(nameToTensor_, feed_dict=feedDict)
        return nameToValue


class FetchTensorValuesAcrossCollections:
    def __init__(self, model, rename):
        self.model = model
        self.rename = rename

    def __call__(self, keyword, feedDict=None):
        nameToTensor_ = dict()
        g = self.model.graph
        collectionNames = g.get_all_collection_keys()
        for collectionName in collectionNames:
            if keyword in collectionName:
                t_ = g.get_collection(collectionName)[0]
                nameToTensor_[self.rename(collectionName)] = t_
        nameToValue = self.model.run(nameToTensor_, feed_dict=feedDict)
        return nameToValue


class PlotHistograms:
    def __init__(self, figSize, summarizeStats, useLogScale=False, plotConfig=None, spaceBtwSubplots=(0.4, 0.6)):
        self.figSize = figSize
        self.summarizeStats = summarizeStats
        self.useLogScale = useLogScale
        self.plotConfig = plotConfig
        self.wspace, self.hspace = spaceBtwSubplots

    def __call__(self, valueName, valueDict, savePath):
        valueDict = {name: value.flatten() for name, value in valueDict.items()}

        layerCount = {"shared": 0, "action": 0, "value": 0}
        for name, value in valueDict.items():
            for layerName in layerCount:
                if layerName in name:
                    layerCount[layerName] += 1
        plotRows = len(layerCount)
        plotCols = max(list(layerCount.values()))
        fig = plt.figure(figsize=self.figSize)
        gs = fig.add_gridspec(plotRows, plotCols)

        allValue = np.concatenate(list(valueDict.values()))
        if self.useLogScale:
            logAllValue = np.log10(allValue + 1e-10)
            _, logBins = np.histogram(logAllValue, bins="auto")
            bins = 10 ** logBins
        else:
            _, bins = np.histogram(allValue, bins="auto")

        layerNameToPos = {"shared": [0, 0], "action": [1, 0], "value": [2, 0]}
        axs = []
        for name, value in valueDict.items():
            for layerName in layerNameToPos:
                if layerName in name:
                    r, c = layerNameToPos[layerName]
                    layerNameToPos[layerName] = [r, c + 1]
            ax = fig.add_subplot(gs[r, c])
            axs.append(ax)
            summary = self.summarizeStats(value)
            ax.set_title("{}\n$\mu=${:.2E} $\sigma=${:.2E}".format(name, summary["mean"], summary["std"]))
            if self.useLogScale:
                ax.set_xscale("log")
                logValue = np.log10(value + 1e-10)
                counts, _ = np.histogram(logValue, bins=logBins)
            else:
                counts, _ = np.histogram(value, bins=bins)
            ax.hist(bins[:-1], bins=bins, weights=counts / np.sum(counts))

        xMin, xMax = np.inf, -np.inf
        yMin, yMax = np.inf, -np.inf
        for ax in axs:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            xMin = min(x0, xMin)
            xMax = max(x1, xMax)
            yMin = min(y0, yMin)
            yMax = max(y1, yMax)
        for ax in axs:
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)

        plt.suptitle("Histograms of {}".format(valueName))
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)

        if savePath is not None:
            plt.savefig(savePath)
        plt.show()


class PlotBoxes:
    def __init__(self, figSize, useLogScale=False):
        self.figSize = figSize
        self.useLogScale = useLogScale

    def __call__(self, valueName, valueDict, savePath):
        fig, ax = plt.subplots(figsize=self.figSize)
        data = list(valueDict.values())
        labels = list(valueDict.keys())
        if self.useLogScale:
            ax.set_yscale('log')
        ax.boxplot(data, labels=labels)
        ax.set_title("Box plots of {}".format(valueName))
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        if savePath is not None:
            plt.savefig(savePath)
        plt.show()


def main():
    numStateSpace = 4
    numActionSpace = 8
    generateModel = net.GenerateModel(numStateSpace, numActionSpace)
    sharedWidths = [100]
    actionWidths = [300, 300, 200]
    valueWidths = [200]
    policyValueNet = generateModel(sharedWidths, actionWidths, valueWidths)
    fetchTensorValuesInOneCollection = FetchTensorValuesInOneCollection(policyValueNet, clipTensorName)
    fetchTensorValuesAcrossCollections = FetchTensorValuesAcrossCollections(policyValueNet, clipTensorName)

    dataSetsDir = '../../data/compareValueDataStandardizationAndLossCoefs/trainingData/dataSets'
    extension = ".pickle"
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

    g = policyValueNet.graph
    states_ = g.get_collection("inputs")[0]
    gtActions_, gtValues_ = g.get_collection("groundTruths")
    feedDict = {states_: trainData[0], gtActions_: trainData[1], gtValues_: trainData[2]}

    keyword = "grad"
    valueDict = fetchTensorValuesAcrossCollections(keyword, feedDict)

    weightGradientValueDict = dict()
    for name in valueDict:
        if "kernel" in name:
            weightGradientValueDict[name] = valueDict[name]

    biasGradientValueDict = dict()
    for name in valueDict:
        if "bias" in name:
            biasGradientValueDict[name] = valueDict[name]

    flattenedValueDict = {name: value.flatten() for name, value in weightGradientValueDict.items()}
    absValueDict = {name: np.abs(value) for name, value in flattenedValueDict.items()}

    figSize = (15, 10)
    useLogScale = True
    plotBoxes = PlotBoxes(figSize, useLogScale)
    plotHists = PlotHistograms(figSize, summarizeStats, useLogScale)

    boxPlotSavePath = None
    histPlotSavePath = None
    plotHists("weight gradients", absValueDict, histPlotSavePath)


if __name__ == "__main__":
    main()
