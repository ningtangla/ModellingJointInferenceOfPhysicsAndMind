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
    def __init__(self, figSize, summarizeStats, plotConfig=None, spaceBtwSubplots=(0.4, 0.6)):
        self.figSize = figSize
        self.summarizeStats = summarizeStats
        self.plotConfig = plotConfig
        self.wspace, self.hspace = spaceBtwSubplots

    def __call__(self, valueName, valueDict, savePath):
        numOfSubplots = len(valueDict)
        if self.plotConfig is not None:
            plotCols, plotRows = self.plotConfig
        else:
            plotCols = int(np.ceil(np.sqrt(numOfSubplots)))
            plotRows = int(np.ceil(numOfSubplots / plotCols))
        fig, axs = plt.subplots(plotRows, plotCols, figsize=self.figSize)

        axs = axs.flat
        for ax in axs[len(valueDict):]:
            ax.remove()
        axs = axs[:len(valueDict)]

        for ax, (name, value) in zip(axs, valueDict.items()):
            summary = self.summarizeStats(value)
            ax.set_title("{}\n$\mu=${:.4f} $\sigma=${:.4f}\nrange=[{:.4f}, {:.4f}]"
                         .format(name, summary["mean"], summary["std"], summary["min"], summary["max"]))
            ax.hist(value.flatten(), density=1)
        plt.suptitle("Histograms of {}".format(valueName))
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        if savePath is not None:
            plt.savefig(savePath)
        plt.show()


def main():
    numStateSpace = 4
    numActionSpace = 8
    generateModel = net.GenerateModel(numStateSpace, numActionSpace)
    sharedWidths = [8]
    actionWidths = [8, 8, 4]
    valueWidths = [4]
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


    keyword = "gradient"
    valueDict = fetchTensorValuesAcrossCollections(keyword, feedDict)

    figSize = (15, 10)
    plotHists = PlotHistograms(figSize, summarizeStats)

    savePath = None
    plotHists(keyword + 's', valueDict, savePath)


if __name__ == "__main__":
    main()
