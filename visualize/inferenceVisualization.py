import numpy as np
import pygame as pg
import pandas as pd
import os
from pylab import plt

class SaveImage:
    def __init__(self, imageFolderName):
        self.imageFolderName = imageFolderName

    def __call__(self, screen):
        currentDir = os.getcwd()
        parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
        saveImageDir = os.path.join(os.path.join(parentDir, 'demo'), self.imageFolderName)
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        fileCount = len([name for name in os.listdir(saveImageDir) if os.path.isfile(os.path.join(saveImageDir, name))])
        pg.image.save(screen, saveImageDir + '/' + format(fileCount, '05') + ".png")


class GetChasingRoleColor:
    def __init__(self, roleColor, roleIndex):
        self.roleColor = roleColor
        self.roleIndex = roleIndex

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        indexList = resultGroupedByChasingIndices['mind']
        chasingFilter = [chasingCondition[agentIndex] == self.roleIndex for chasingCondition in indexList]
        chasingIndexList = [i for i, index in zip(range(len(indexList)), chasingFilter) if index]
        agentPosterior = sum(
            [resultGroupedByChasingIndices.iloc[chasingIndex]['posterior'] for chasingIndex in
             chasingIndexList])
        newColor = tuple(agentPosterior * np.array(self.roleColor))
        return newColor


class GetChasingResultColor:
    def __init__(self, getRolesColor):
        self.getRolesColor = getRolesColor

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        rolesColor = [getColor(agentIndex, resultGroupedByChasingIndices) for getColor in self.getRolesColor]
        resultColor = tuple(sum(np.array(rolesColor)))
        return resultColor


class ColorChasingPoints:
    def __init__(self, getChasingResultColor):
        self.getChasingResultColorColor = getChasingResultColor

    def __call__(self, resultGroupedByChasingIndices):
        agentCount = len(resultGroupedByChasingIndices['mind'][0])
        colorList = [self.getChasingResultColorColor(agentIndex, resultGroupedByChasingIndices) for agentIndex in
                     range(agentCount)]
        return colorList


class AdjustPullingLineWidth:
    def __init__(self, minWidth = 1, maxWidth = 5):
        self.minWidth = minWidth
        self.maxWidth = maxWidth

    def __call__(self, resultGroupedByPullingIndices):
        pullingIndexList = resultGroupedByPullingIndices['physics']
        lineWidthDict = dict()
        for index in range(len(pullingIndexList)):
            pullingIndex = pullingIndexList[index]
            agentPosterior = resultGroupedByPullingIndices['posterior'].values[index]

            pullingWidth = int(np.round(self.minWidth + agentPosterior * (self.maxWidth - self.minWidth)))
            pulledAgentID, pullingAgentID = np.where(np.array(pullingIndex) == 'pulled')[0]

            lineWidthDict[(pulledAgentID, pullingAgentID)] = pullingWidth

        return lineWidthDict


class DrawInferenceResultWithPull:
    def __init__(self, inferenceIndex, gridPixelSize, drawGrid, drawCirclesAndLines,
                 colorChasingPoints, adjustPullingLineWidth):
        self.inferenceIndex = inferenceIndex
        
        self.gridPixelSize = gridPixelSize
        self.drawGrid = drawGrid
        self.drawCirclesAndLines = drawCirclesAndLines

        self.colorChasingPoints = colorChasingPoints
        self.adjustPullingLineWidth = adjustPullingLineWidth

    def __call__(self, state, posterior):
        screen = self.drawGrid()
        pointsCoord = state
        pointsLocation = [list(np.array(pointCoord) * self.gridPixelSize - self.gridPixelSize // 2)
                          for pointCoord in pointsCoord]
        
        inferenceDf = pd.DataFrame(index= self.inferenceIndex)
        inferenceDf['posterior'] = posterior
        
        resultGroupedByChasingIndices = inferenceDf.groupby('mind').sum().reset_index()
        pointsColor = self.colorChasingPoints(resultGroupedByChasingIndices)

        resultGroupedByPullingIndices = inferenceDf.groupby('physics').sum().reset_index()
        linesWidth = self.adjustPullingLineWidth(resultGroupedByPullingIndices)

        screen = self.drawCirclesAndLines(screen, pointsLocation, pointsColor, linesWidth)

        return screen


class DrawContinuousInferenceResultNoPull:
    def __init__(self, numOfAgents, inferenceIndex, drawState, scaleState,
                 colorChasingPoints, adjustFPS, saveImage):
        self.numOfAgents = numOfAgents
        self.inferenceIndex = inferenceIndex
        self.drawState = drawState
        self.scaleState = scaleState
        self.colorChasingPoints = colorChasingPoints
        self.adjustFPS = adjustFPS
        self.saveImage = saveImage

    def __call__(self, currentState, nextState, posterior):
        currentPosition = self.scaleState(currentState)
        nextPosition =  self.scaleState(nextState)
        print("currentPosition: ", currentPosition)

        inferenceDf = pd.DataFrame(index=self.inferenceIndex)
        inferenceDf['posterior'] = posterior

        resultGroupedByChasingIndices = inferenceDf.groupby('mind').sum().reset_index()
        circleColorList = self.colorChasingPoints(resultGroupedByChasingIndices)

        positionsList = self.adjustFPS(currentPosition, nextPosition)

        for positionIndex in range(len(positionsList)):
            screen = self.drawState(self.numOfAgents, positionsList[positionIndex], circleColorList)
            if self.saveImage is not None:
                self.saveImage(screen)


class PlotInferenceProb:
    def __init__(self, xVariableName, yVaraibleName, groupIndex):
        self.xVariableName = xVariableName
        self.yVaraibleName = yVaraibleName
        self.groupIndex = groupIndex

    def __call__(self, inferenceDf, graphIndex, plotname):
        # print(inferenceDf)
        resultDf = inferenceDf.groupby(self.groupIndex).sum()
        print(resultDf)
        graph = resultDf.T.plot()
        graph.set_xlabel(self.xVariableName)
        graph.set_ylabel(self.yVaraibleName)
        plt.ylim([0, 1])
        plt.title(self.groupIndex+ ": " + plotname + '_dataIndex'+ str(graphIndex))
        dirName = os.path.dirname(__file__)
        plotPath = os.path.join(dirName, '..', 'demo')
        plt.savefig(os.path.join(plotPath, self.groupIndex + plotname + 'data'+ str(graphIndex)))
        plt.show()
