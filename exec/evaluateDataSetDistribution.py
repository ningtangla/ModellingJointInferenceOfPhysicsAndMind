import sys
import os
src = os.path.join(os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import pandas as pd
import pickle
from pylab import plt
from collections import OrderedDict
from evaluationFunctions import GetSavePath


class GetAgentStateFromDataSetState:

    def __init__(self, agentStateDim):
        self.agentStateDim = agentStateDim

    def __call__(self, dataSetState, agentID):
        indexes = [
            agentID * self.agentStateDim + num
            for num in range(self.agentStateDim)
        ]
        agentState = [dataSetState[index] for index in indexes]
        return agentState


def isin(range, number):
    left, right = range
    if right > number > left:
        return True
    return False


class ComputeDistribution:

    def __init__(self, statistic, stateKey, getAgentStateFromDataSetState,
                 getSavePathForDataSet):
        self.statistic = statistic
        self.stateKey = stateKey
        self.getAgentStateFromDataSetState = getAgentStateFromDataSetState
        self.getSavePathForDataSet = getSavePathForDataSet

    def __call__(self, df):
        xSection = df.index.get_level_values('xSection')[0]
        ySection = df.index.get_level_values('ySection')[0]
        agentID = df.index.get_level_values('agentID')[0]
        augmentedParameter = {
            'augmented': df.index.get_level_values('augmented')[0]
        }
        dataSetPath = self.getSavePathForDataSet(augmentedParameter)
        if not os.path.exists(dataSetPath):
            print("No such DataSet {}".format(dataSetPath))
            return
        with open(dataSetPath, 'rb') as f:
            dataSet = pickle.load(f)
        dataSetStates = dataSet[self.stateKey]
        agentStates = [
            self.getAgentStateFromDataSetState(state, agentID)
            for state in dataSetStates
        ]
        inSectionStates = [
            state for state in agentStates
            if isin(xSection, state[0]) and isin(ySection, state[1])
        ]
        return pd.Series({self.statistic: len(inSectionStates)})


def main():
    # data
    dataDir = os.path.join(os.pardir, 'data', 'evaluateDataSetDistribution')
    trainingDataDir = os.path.join(dataDir, "dataSets")
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
    getSavePathForDataSet = GetSavePath(trainingDataDir, dataSetExtension,
                                        dataSetParameter)

    # env
    levelNames = ["xSection", "ySection", "agentID", "augmented"]
    discreteFactor = 10
    xBoundary = [0, 180]
    xSections = [(xBoundary[1] / discreteFactor * cnt,
                  xBoundary[1] / discreteFactor * (cnt + 1))
                 for cnt in range(discreteFactor)]
    yBoundary = [0, 180]
    ySections = [(yBoundary[1] / discreteFactor * cnt,
                  yBoundary[1] / discreteFactor * (cnt + 1))
                 for cnt in range(discreteFactor)]
    sheepID = 0
    wolfID = 1
    agentID = [wolfID, sheepID]
    augmented = ['True', 'False']
    levelValues = [xSections, ySections, agentID, augmented]
    MultiIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=MultiIndex)
    statistic = "num"
    agentStateDim = 2
    getAgentStateFromDataSetState = GetAgentStateFromDataSetState(agentStateDim)
    stateKey = 'state'
    computeDistribution = ComputeDistribution(statistic, stateKey,
                                              getAgentStateFromDataSetState,
                                              getSavePathForDataSet)
    statDF = toSplitFrame.groupby(levelNames).apply(computeDistribution)

    # draw
    xName = "xSection"
    yName = "ySection"
    subplotRowName = "agentID"
    subplotColName = "augmented"
    rowNum = len(agentID)
    colNum = len(augmented)
    colName = statistic
    drawingExtent = tuple(xBoundary) + tuple(yBoundary)
    figsize = (12, 10)
    figure = plt.figure(figsize=figsize)
    numOfPlot = 1
    agentIDNameDict = {wolfID: 'wolf', sheepID: 'sheep'}
    for colKey, subDF in statDF.groupby(subplotColName):
        colorBarMaxLimit = max(subDF[statistic])
        for rowKey, agentDF in subDF.groupby(subplotRowName):
            ax = figure.add_subplot(rowNum, colNum, numOfPlot)
            resetDF = agentDF.reset_index().sort_values(yName, ascending="True")
            plotDF = resetDF.pivot(index=xName, columns=yName, values=colName)
            cValues = plotDF.values
            xticks = [coor[0] for coor in plotDF.columns.values]
            yticks = [coor[0] for coor in plotDF.index.values]
            newDf = pd.DataFrame(cValues, index=yticks, columns=xticks)
            image = ax.imshow(newDf,
                              vmin=0,
                              vmax=colorBarMaxLimit,
                              origin="lower",
                              extent=drawingExtent)
            plt.colorbar(image, fraction=0.046, pad=0.04)
            agentIDName = agentIDNameDict[rowKey]
            if colKey == 'True':
                augmentedTitle = 'augmented'
            else:
                augmentedTitle = ''
            subtitle = "{} {} distribution".format(augmentedTitle, agentIDName)
            plt.title(subtitle)
            numOfPlot += 1
    title = "DataSet Distribution"
    plt.suptitle(title)
    graphDir = os.path.join(dataDir, "Graphs")
    if not os.path.exists(graphDir):
        os.mkdir(graphDir)
    getSavePathForGraph = GetSavePath(graphDir, ".png")
    graphPath = getSavePathForGraph(dataSetParameter)
    plt.savefig(graphPath)


if __name__ == "__main__":
    main()
