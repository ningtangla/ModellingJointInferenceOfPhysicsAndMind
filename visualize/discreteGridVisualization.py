import numpy as np
from itertools import combinations
import pygame as pg
import sys

def checkDuplicates(checkingList):
    itemCount = len(checkingList)
    checkingPairs = list(combinations(range(itemCount), 2))
    isSamePair = lambda pairOne, pairTwo: np.all(np.array(pairOne) == np.array(pairTwo))
    checkingFilter = [isSamePair(checkingList[i], checkingList[j]) for i, j in checkingPairs]
    duplicatePairs = [pair for pair, index in zip(checkingPairs, checkingFilter) if index]
    return duplicatePairs


class ModifyOverlappingPoints:
    def __init__(self, gridPixelSize, checkDuplicates, modificationRatio):
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


class DrawCircles:
    def __init__(self, colorList, modifyOverlappingPoints, pointWidth = 10,  pointExtendTime = 100, FPS = 60):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.pointWidth = pointWidth
        self.colorList = colorList
        self.modifyOverlappingPoints = modifyOverlappingPoints
        self.drawPoint = lambda game, color, point: pg.draw.circle(game, color, point, self.pointWidth)

    def __call__(self, screen, pointsLocation):
        for frameNumber in range(self.pointExtendFrame):
            pointsLocation = self.modifyOverlappingPoints(pointsLocation)

            for agentIndex in range(len(self.colorList)):
                self.drawPoint(screen, self.colorList[agentIndex], pointsLocation[agentIndex])

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            pg.display.update()
        return screen




class DrawGrid:
    def __init__(self, game, gridSize, gridPixelSize, backgroundColor = (255, 255, 255), gridColor = (0,0,0), gridLineWidth = 3):
        self.game = game
        self.gridNumberX, self.gridNumberY = gridSize
        self.gridPixelSize = gridPixelSize
        self.backgroundColor = backgroundColor
        self.gridColor = gridColor
        self.gridLineWidth = gridLineWidth

    def __call__(self):
        upperBoundX = self.gridPixelSize * self.gridNumberX
        upperBoundY = self.gridPixelSize * self.gridNumberY
        self.game.fill(self.backgroundColor)

        for gridIndexX in range(self.gridNumberX + 1):
            gridX = int(gridIndexX * self.gridPixelSize)
            pg.draw.line(self.game, self.gridColor, (gridX, 0), (gridX, upperBoundY), self.gridLineWidth)

        for gridIndexY in range(self.gridNumberY + 1):
            gridY = int(gridIndexY * self.gridPixelSize)
            pg.draw.line(self.game, self.gridColor, (0, gridY), (upperBoundX, gridY), self.gridLineWidth)
        return self.game



class DrawCirclesAndLines:
    def __init__(self, modifyOverlappingPoints, pointExtendTime = 100, FPS = 60, circleSize = 10, lineColor = (0,0,0)):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.circleSize = circleSize
        self.modifyOverlappingPoints = modifyOverlappingPoints
        self.lineColor = lineColor
        self.drawPoint = lambda game, color, point: pg.draw.circle(game, color, point, self.circleSize)

    def __call__(self, screen, pointsLocation, colorList, lineWidthDict):
        for frameNumber in range(self.pointExtendFrame):
            pointsLocation = self.modifyOverlappingPoints(pointsLocation)

            for agentIndex in range(len(colorList)):
                self.drawPoint(screen, colorList[agentIndex], pointsLocation[agentIndex])

            for lineConnectedAgents in lineWidthDict.keys():
                pullingAgentIndex, pulledAgentIndex = lineConnectedAgents
                pullingAgentLoc = pointsLocation[pullingAgentIndex]
                pulledAgentLoc = pointsLocation[pulledAgentIndex]
                lineWidth = lineWidthDict[lineConnectedAgents]
                pg.draw.line(screen, self.lineColor, pullingAgentLoc, pulledAgentLoc, lineWidth)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

            pg.display.flip()
        return screen


class DrawPointsAndSaveImage:
    def __init__(self, drawGrid, drawCircles, gridPixelSize, fps = 60):
        self.fps = fps
        self.drawGrid = drawGrid
        self.drawCircles = drawCircles
        self.gridPixelSize = gridPixelSize

    def __call__(self, trajectory, iterationNumber, saveImage = False):
        fpsClock = pg.time.Clock()
        for timeStep in range(len(trajectory)):
            state = trajectory[timeStep]
            fpsClock.tick(self.fps)
            game = self.drawGrid()
            pointsLocation = [list (np.array(agentState) * self.gridPixelSize - self.gridPixelSize//2)
                                  for agentState in state]
            game = self.drawCircles(game, pointsLocation)
            if saveImage:
                pg.image.save(game, "screenshot"+ format(timeStep, '04') + ".png")
        return

