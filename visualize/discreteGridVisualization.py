import numpy as np 
from itertools import combinations
import pygame 



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
        self.modifyPointAdd = lambda pointLocation: np.array(pointLocation) + self.gridPixelSize // self.modificationRatio
        self.modifyPointMinus = lambda pointLocation: np.array(pointLocation) - self.gridPixelSize // self.modificationRatio
        self.modifyPoints = lambda firstCoor, secondCoor: [self.modifyPointAdd(firstCoor), self.modifyPointMinus(secondCoor)]

    def __call__(self, pointsLocation):
        duplicatePointPairs = self.checkDuplicates(pointsLocation)
        if len(duplicatePointPairs) != 0:
            for pairIndex in duplicatePointPairs:
                firstPointIndex, secondPointIndex = pairIndex
                pointsLocation[firstPointIndex], pointsLocation[secondPointIndex] = self.modifyPoints(pointsLocation[firstPointIndex], pointsLocation[secondPointIndex])
        return pointsLocation



class DrawCircles:
    def __init__(self, pointExtendTime, FPS, colorList, pointWidth, modifyOverlappingPoints):
        self.FPS = FPS
        self.pointExtendFrame = int(pointExtendTime * self.FPS / 1000)
        self.pointWidth = pointWidth
        self.colorList = colorList
        self.modifyOverlappingPoints = modifyOverlappingPoints
        self.drawPoint = lambda game, color, point: pygame.draw.circle(game, color, point, self.pointWidth)

    def __call__(self, game, pointsLocation):
        for frameNumber in range(self.pointExtendFrame):
            pointsLocation = self.modifyOverlappingPoints(pointsLocation)

            for agentIndex in range(len(self.colorList)):
                self.drawPoint(game, self.colorList[agentIndex], pointsLocation[agentIndex])

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
        self.backgroundColor=backgroundColor
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


class DrawPointsFromLocationDfAndSaveImage:
    def __init__(self, initializeGame, drawGrid, drawCircles, gridPixelSize):
        self.initializaGame = initializeGame
        self.drawGrid = drawGrid
        self.drawCircles = drawCircles
        self.gridPixelSize = gridPixelSize

    def __call__(self, trajectory, iterationNumber, saveImage = False):
        game = self.initializaGame()
        for frameIndex in range(len(trajectory)):
            game = self.drawGrid(game)
            pointsCoord = trajectory[frameIndex]
            pointsLocation = [list (np.array(pointCoord) * self.gridPixelSize - self.gridPixelSize//2)
                                  for pointCoord in pointsCoord]
            game = self.drawCircles(game, pointsLocation)
            if saveImage:
                pygame.image.save(game, "screenshot"+ format(frameIndex, '04') + ".png")
