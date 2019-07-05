import sys
import os
src = os.path.join('..', 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import pandas as pd
import pickle
from pylab import plt
from collections import OrderedDict
from evaluationFunctions import GetSavePath


class GetAgentStateFromDataSetState():
    def __init__(self, agentID, agentStateDim):
        self.agentID = agentID
        self.agentStateDim = agentStateDim

    def __call__(self, dataSetState):
        indexes = [self.agentID*self.agentStateDim + num
                   for num in range(self.agentStateDim)]
        agentState = [dataSetState[index] for index in indexes]
        return agentState


def isin(range, number):
    left, right = range
    if number > left and number < right:
        return True
    return False


class GenerateStatisticForDistribution():
    def __init__(self, statistic):
        self.statistic = statistic

    def __call__(self, df, states):
        xSection = df.index.get_level_values('xSection')[0]
        ySection = df.index.get_level_values('ySection')[0]
        inSectionStates = [state for state in states
                              if isin(xSection, state[0])
                              and isin(ySection, state[1])]
        return pd.Series({self.statistic: len(inSectionStates)})


class DrawDataDistribution():
    def __init__(self, xName, yName, colName, extent):
        self.xName = xName
        self.yName = yName
        self.colName = colName
        self.counter = 1
        self.extent = extent

    def __call__(self, df, figure, title):
        ax = figure.add_subplot(1, 2, self.counter)
        resetDF = df.reset_index().sort_values(self.yName, ascending="True")
        plotDF = resetDF.pivot(index=self.xName, columns=self.yName,
                               values=self.colName)
        cValues = plotDF.values
        xticks = [coor[0] for coor in plotDF.columns.values]
        yticks = [coor[0] for coor in plotDF.index.values]
        newDf = pd.DataFrame(cValues, index=yticks, columns=xticks)
        image = ax.imshow(newDf, vmin=0, vmax=11000,
                       origin="lower", extent=self.extent)
        plt.colorbar(image, fraction=0.046, pad=0.04)
        plt.title(title)
        self.counter += 1


def main():
    # data
    dataDir = os.path.join('..', 'data', 'evaluateDataSetDistribution')
    trainingDataDir = os.path.join(dataDir, "augmentedDataSets")
    dataSetParameter = OrderedDict()
    dataSetParameter['cBase'] = 100
    dataSetParameter['initPos'] = 'Random'
    dataSetParameter['maxRunningSteps'] = 30
    dataSetParameter['numDataPoints'] = 68181
    dataSetParameter['numSimulations'] = 200
    dataSetParameter['numTrajs'] = 2500
    dataSetParameter['rolloutSteps'] = 10
    dataSetParameter['standardizedReward'] = 'True'
    dataSetExtension = '.pickle'
    getSavePathForDataSet = GetSavePath(trainingDataDir, dataSetExtension)
    dataSetPath = getSavePathForDataSet(dataSetParameter)
    if not os.path.exists(dataSetPath):
        print("No such DataSet {}".format(dataSetPath))
        return
    with open(dataSetPath, 'rb') as f:
        dataSet = pickle.load(f)
    stateKey = "state"
    dataSetStates = dataSet[stateKey]
    wolfID = 1
    sheepID = 0
    agentStateDim = 2
    getSheepState = GetAgentStateFromDataSetState(sheepID, agentStateDim)
    getWolfState = GetAgentStateFromDataSetState(wolfID, agentStateDim)
    sheepStates = [getSheepState(state) for state in dataSetStates]
    wolfStates = [getWolfState(state) for state in dataSetStates]

    # env
    statistic = "num"
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    discreteFactor = 10
    xSections = [(xBoundary[1]/discreteFactor *
                  cnt, xBoundary[1]/discreteFactor*(cnt+1))
                 for cnt in range(discreteFactor)]
    ySections = [(yBoundary[1]/discreteFactor *
                  cnt, yBoundary[1]/discreteFactor*(cnt+1))
                 for cnt in range(discreteFactor)]
    levelNames = ["xSection", "ySection"]
    levelValues = [xSections, ySections]
    MultiIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=MultiIndex)
    generateStatisticForDistribution = GenerateStatisticForDistribution(statistic)
    sheepStatistic = toSplitFrame.groupby(levelNames).apply(generateStatisticForDistribution, sheepStates)
    wolfStatisitc = toSplitFrame.groupby(levelNames).apply(generateStatisticForDistribution, wolfStates)

    drawingExtent = tuple(xBoundary) + tuple(yBoundary)
    figure = plt.figure(figsize=(12, 10))
    drawer = DrawDataDistribution("xSection", "ySection", statistic, drawingExtent)
    drawer(sheepStatistic, figure, "sheepDistribution")
    drawer(wolfStatisitc, figure, "wolfDistribution")

    graphDir = os.path.join(dataDir, "Graphs")
    if not os.path.exists(graphDir):
        os.mkdir(graphDir)
    getSavePathForGraph = GetSavePath(graphDir, ".png")
    graphPath = getSavePathForGraph(dataSetParameter)
    print(graphPath)
    exit()
    plt.savefig(graphPath)


if __name__ == "__main__":
    main()