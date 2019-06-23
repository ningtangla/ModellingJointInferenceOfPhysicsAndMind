from pylab import plt
import pandas as pd
import numpy as np


class DrawHeatMap:
    def __init__(self, groupByVariableNames, subplotIndex, subplotIndexName):
        self.groupByVariableNames = groupByVariableNames
        self.subplotIndex = subplotIndex
        self.subplotIndexName = subplotIndexName

    def __call__(self, dataDF, colName):
        figure = plt.figure(figsize=(12, 10))
        numOfplot = 1
        subDFs = reversed([subDF for key, subDF in dataDF.groupby(self.groupByVariableNames[0])])
        OrderedSubDFs = [df for subDF in subDFs for key, df in subDF.groupby(self.groupByVariableNames[1])]
        for subDF in OrderedSubDFs:
            subplot = figure.add_subplot(self.subplotIndex[0], self.subplotIndex[1], numOfplot)
            plotDF = subDF.reset_index()
            plotDF.plot.scatter(x=self.subplotIndexName[0], y=self.subplotIndexName[1], c=colName, colormap="jet", ax=subplot, vmin=0, vmax=9)
            numOfplot = numOfplot + 1
        plt.subplots_adjust(wspace=0.8, hspace=0.4)
        plt.show()


def simpleApply(df):
    index = df.index[0]
    if (np.array(index[0:2]) == np.array(index[2:4])).all():
        return pd.Series({"value": 0})
    else:
        return pd.Series({"value": 6})


if __name__ == '__main__':
    discreteFactor = 4
    Boundary = 180
    discreteRange = Boundary / (discreteFactor + 1)
    xPosition = [discreteRange * (i + 1) for i in range(discreteFactor)]
    yPosition = xPosition
    levelNames = ["InX", "InY", "OutX", "OutY"]
    levelValues = [xPosition, yPosition, xPosition, yPosition]
    diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=diffIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(simpleApply)
    draw = DrawHeatMap(['OutY', 'OutX'], [len(yPosition), len(xPosition)],
                ['InX', 'InY'])
    draw(resultDF, "value")
