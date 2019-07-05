import numpy as np
from itertools import combinations
import pygame
import sys


def checkDuplicates(checkingList):
    itemCount = len(checkingList)
    checkingPairs = list(combinations(range(itemCount), 2))
    isSamePair = lambda pairOne, pairTwo: np.all(np.array(pairOne) == np.array(pairTwo))
    checkingFilter = [isSamePair(checkingList[i], checkingList[j]) for i, j in checkingPairs]
    duplicatePairs = [pair for pair, index in zip(checkingPairs, checkingFilter) if index]
    return duplicatePairs


class ModifyOverlappingPoints:
    def __init__(self, gridPixelSize, modificationRatio, checkDuplicates):
        self.gridPixelSize = gridPixelSize
        self.modificationRatio = modificationRatio
        self.checkDuplicates = checkDuplicates
        self.modifyPointAdd = lambda pointLocation: np.array(
            pointLocation) + self.gridPixelSize // self.modificationRatio
        self.modifyPointMinus = lambda pointLocation: np.array(
            pointLocation) - self.gridPixelSize // self.modificationRatio
        self.modifyPoints = lambda firstCoor, secondCoor: [self.modifyPointAdd(firstCoor),
                                                           self.modifyPointMinus(secondCoor)]

    def __call__(self, pointsLocation):
        duplicatePointPairs = self.checkDuplicates(pointsLocation)
        if len(duplicatePointPairs) != 0:
            for pairIndex in duplicatePointPairs:
                firstPointIndex, secondPointIndex = pairIndex
                pointsLocation[firstPointIndex], pointsLocation[secondPointIndex] = self.modifyPoints(
                    pointsLocation[firstPointIndex], pointsLocation[secondPointIndex])
        return pointsLocation


class DrawCirclesAndLines:
    def __init__(self, pointExtendTime, FPS, pointWidth, lineColor, modifyOverlappingPoints):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.pointWidth = pointWidth
        self.modifyOverlappingPoints = modifyOverlappingPoints
        self.lineColor = lineColor
        self.drawPoint = lambda game, color, point: pygame.draw.circle(game, color, point, self.pointWidth)

    def __call__(self, game, pointsLocation, colorList, lineWidthDict):
        for frameNumber in range(self.pointExtendFrame):
            pointsLocation = self.modifyOverlappingPoints(pointsLocation)

            for agentIndex in range(len(colorList)):
                self.drawPoint(game, colorList[agentIndex], pointsLocation[agentIndex])

            for lineConnectedAgents in lineWidthDict.keys():
                pullingAgentIndex, pulledAgentIndex = lineConnectedAgents
                pullingAgentLoc = pointsLocation[pullingAgentIndex]
                pulledAgentLoc = pointsLocation[pulledAgentIndex]
                lineWidth = lineWidthDict[lineConnectedAgents]
                pygame.draw.line(game, self.lineColor, pullingAgentLoc, pulledAgentLoc, lineWidth)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
        return game


class InitializeGame:
    def __init__(self, screenWidth, screenHeight, caption):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.caption = caption

    def __call__(self):
        pygame.init()
        game = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption(self.caption)
        return game


class DrawGrid:
    def __init__(self, gridSize, gridPixelSize, backgroundColor, gridColor, gridLineWidth):

        self.gridNumberX, self.gridNumberY = gridSize
        self.gridPixelSize = gridPixelSize
        self.backgroundColor = backgroundColor
        self.gridColor = gridColor
        self.gridLineWidth = gridLineWidth

    def __call__(self, game):
        upperBoundX = self.gridPixelSize * self.gridNumberX
        upperBoundY = self.gridPixelSize * self.gridNumberY
        game.fill(self.backgroundColor)

        for gridIndexX in range(self.gridNumberX + 1):
            gridX = int(gridIndexX * self.gridPixelSize)
            pygame.draw.line(game, self.gridColor, (gridX, 0), (gridX, upperBoundY), self.gridLineWidth)

        for gridIndexY in range(self.gridNumberY + 1):
            gridY = int(gridIndexY * self.gridPixelSize)
            pygame.draw.line(game, self.gridColor, (0, gridY), (upperBoundX, gridY), self.gridLineWidth)
        return game


class GetChasingRoleColor:
    def __init__(self, roleColor, roleIndex):
        self.roleColor = roleColor
        self.roleIndex = roleIndex

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        indexList = resultGroupedByChasingIndices['chasingAgents']
        chasingFilter = [chasingCondition[agentIndex] == self.roleIndex for chasingCondition in indexList]
        chasingIndexList = [i for i, index in zip(range(len(indexList)), chasingFilter) if index]
        agentPosterior = sum(
            [resultGroupedByChasingIndices.iloc[chasingIndex]['normalizedPosterior'] for chasingIndex in
             chasingIndexList])
        newColor = tuple(agentPosterior * np.array(self.roleColor))
        return newColor


class GetChasingResultColor:
    def __init__(self, getWolfColor, getSheepColor, getMasterColor):
        self.getWolfColor = getWolfColor
        self.getSheepColor = getSheepColor
        self.getMasterColor = getMasterColor

    def __call__(self, agentIndex, resultGroupedByChasingIndices):
        wolfColor = self.getWolfColor(agentIndex, resultGroupedByChasingIndices)
        sheepColor = self.getSheepColor(agentIndex, resultGroupedByChasingIndices)
        masterColor = self.getMasterColor(agentIndex, resultGroupedByChasingIndices)
        resultColor = tuple(np.array(wolfColor) + np.array(sheepColor) + np.array(masterColor))
        return resultColor


class ColorChasingPoints:
    def __init__(self, getChasingResultColor):
        self.getChasingResultColorColor = getChasingResultColor

    def __call__(self, resultGroupedByChasingIndices):
        agentCount = len(resultGroupedByChasingIndices['chasingAgents'][0])
        colorList = [self.getChasingResultColorColor(agentIndex, resultGroupedByChasingIndices) for agentIndex in
                     range(agentCount)]
        return colorList


class AdjustPullingLineWidth:
    def __init__(self, minWidth, maxWidth):
        self.minWidth = minWidth
        self.maxWidth = maxWidth

    def __call__(self, resultGroupedByPullingIndices):
        pullingIndexList = resultGroupedByPullingIndices['pullingAgents']
        lineWidthDict = dict()
        for index in range(len(pullingIndexList)):
            pullingIndex = pullingIndexList[index]
            agentPosterior = resultGroupedByPullingIndices['normalizedPosterior'].values[index]

            pullingWidth = int(np.round(self.minWidth + agentPosterior * (self.maxWidth - self.minWidth)))
            pulledAgentID, pullingAgentID = np.where(np.array(pullingIndex) == 'pulled')[0]

            lineWidthDict[(pulledAgentID, pullingAgentID)] = pullingWidth

        return lineWidthDict


class DrawInferenceResult:
    def __init__(self, gridPixelSize, initializeGame, drawGrid, drawCirclesAndLines,
                 colorChasingPoints, adjustPullingLineWidth):
        self.gridPixelSize = gridPixelSize
        self.initializaGame = initializeGame
        self.drawGrid = drawGrid
        self.drawCirclesAndLines = drawCirclesAndLines

        self.colorChasingPoints = colorChasingPoints
        self.adjustPullingLineWidth = adjustPullingLineWidth

    def __call__(self, screenShotIndex, state, inferenceDf, saveImage=False):
        game = self.initializaGame()
        game = self.drawGrid(game)
        pointsCoord = state
        pointsLocation = [list(np.array(pointCoord) * self.gridPixelSize - self.gridPixelSize // 2)
                          for pointCoord in pointsCoord]

        resultGroupedByChasingIndices = inferenceDf.groupby('chasingAgents').sum().reset_index()
        pointsColor = self.colorChasingPoints(resultGroupedByChasingIndices)

        resultGroupedByPullingIndices = inferenceDf.groupby('pullingAgents').sum().reset_index()
        linesWidth = self.adjustPullingLineWidth(resultGroupedByPullingIndices)

        game = self.drawCirclesAndLines(game, pointsLocation, pointsColor, linesWidth)

        if saveImage:
            pygame.image.save(game, "screenshot" + format(screenShotIndex, '04') + ".png")

