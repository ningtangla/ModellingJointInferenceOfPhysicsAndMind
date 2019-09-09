import pygame as pg
import os
import numpy as np


class ScaleState:
    def __init__(self, positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange):
        self.xIndex, self.yIndex = positionIndex
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange

        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange

    def __call__(self, originalState):
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)

        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (rawY - self.rawYMin) * yScale + self.scaledYMin

        adjustState = lambda state: [adjustX(state[self.xIndex]), adjustY(state[self.yIndex])]

        newState = [adjustState(agentState) for agentState in originalState]

        return newState


class ScaleTrajectory:
    def __init__(self, positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange):
        self.xIndex, self.yIndex = positionIndex
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange

        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange

    def __call__(self, originalTraj):
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)

        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (rawY - self.rawYMin) * yScale + self.scaledYMin

        adjustPair = lambda pair: [adjustX(pair[0]), adjustY(pair[1])]
        agentCount = len(originalTraj[0])

        adjustState = lambda state: [adjustPair(state[agentIndex]) for agentIndex in range(agentCount)]
        trajectory = [adjustState(state) for state in originalTraj]

        return trajectory


class AdjustDfFPStoTraj:
    def __init__(self, oldFPS, newFPS):
        self.oldFPS = oldFPS
        self.newFPS = newFPS

    def __call__(self, trajectory):
        agentNumber = len(trajectory[0])
        xValue = [[state[agentIndex][0] for state in trajectory] for agentIndex in range(agentNumber)]
        yValue = [[state[agentIndex][1] for state in trajectory] for agentIndex in range(agentNumber)]

        timeStepsNumber = len(trajectory)
        adjustRatio = self.newFPS // (self.oldFPS - 1)

        insertPositionValue = lambda positionList: np.array(
            [np.linspace(positionList[index], positionList[index + 1], adjustRatio, endpoint=False)
             for index in range(timeStepsNumber - 1)]).flatten().tolist()
        newXValue = [insertPositionValue(agentXPos) for agentXPos in xValue]
        newYValue = [insertPositionValue(agentYPos) for agentYPos in yValue]

        newTimeStepsNumber = len(newXValue[0])
        getSingleState = lambda time: [(newXValue[agentIndex][time], newYValue[agentIndex][time]) for agentIndex in range(agentNumber)]
        newTraj = [getSingleState(time) for time in range(newTimeStepsNumber)]
        return newTraj


class AdjustStateFPS:
    def __init__(self, oldFPS, newFPS):
        self.oldFPS = oldFPS
        self.newFPS = newFPS

    def __call__(self, currentPosition, nextPosition):
        adjustRatio = self.newFPS // (self.oldFPS - 1)
        positionList = np.linspace(currentPosition, nextPosition, adjustRatio, endpoint=False)
        return positionList


class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        return

class DrawBackgroundWithObstacles:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.allObstaclePos = allObstaclePos
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        [pg.draw.rect(self.screen, self.lineColor, obstaclePos, self.lineWidth) for obstaclePos in
         self.allObstaclePos]

        return


class DrawState:
    def __init__(self, screen, circleSize, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround

    def __call__(self, numOfAgent, state, circleColorList):
        self.drawBackGround()
        for agentIndex in range(numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        return self.screen

class DrawRope():
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, ropePartIndex, ropeColor, ropeWidth, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.ropePartIndex = ropePartIndex
        self.ropeColor = ropeColor
        self.ropeWidth = ropeWidth
        self.drawBackGround = drawBackGround

    def __call__(self, state, tiedAgentId, ropeWidthRatio):
        if tiedAgentId:
            tiedAgentPos = [[np.int(state[agentId][self.xIndex]), np.int(state[agentId][self.yIndex])] for agentId in tiedAgentId]
            ropePosList = [[np.int(state[ropeId][self.xIndex]), np.int(state[ropeId][self.yIndex])] for ropeId in self.ropePartIndex]

            tiedPosList = [[ropePosList[i], ropePosList[i + 1]] for i in range(0, len(ropePosList) - 1)]
            tiedPosList.insert(0, [tiedAgentPos[0], ropePosList[0]])
            tiedPosList.insert(-1, [tiedAgentPos[1], ropePosList[-1]])

            [pg.draw.lines(self.screen, self.ropeColor, False, tiedPos, int(self.ropeWidth * ropeWidthRatio)) for tiedPos in tiedPosList]

        pg.display.flip()
        pg.time.wait(1)

        return

class DrawStateWithRopeInProbability():
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, ropePartIndex, ropeColor, ropeWidth, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.ropePartIndex = ropePartIndex
        self.ropeColor = ropeColor
        self.ropeWidth = ropeWidth
        self.drawBackGround = drawBackGround

    def __call__(self, states, tiedAgentIdPairs, circleColorList, ropeWidthRatios = [1, 1]):
        self.drawBackGround()
        for state, tiedAgentId, ropeWidthRatio in zip(states, tiedAgentIdPairs, ropeWidthRatios):
            tiedAgentPos = [[np.int(state[agentId][self.xIndex]), np.int(state[agentId][self.yIndex])] for agentId in tiedAgentId]
            ropePosList = [[np.int(state[ropeId][self.xIndex]), np.int(state[ropeId][self.yIndex])] for ropeId in self.ropePartIndex]
            tiedPosList = [[ropePosList[i], ropePosList[i + 1]] for i in range(0, len(ropePosList) - 1)]
            if 2 not in tiedAgentId:
                tiedPosList.insert(0, [tiedAgentPos[1], ropePosList[0]])
                tiedPosList.insert(-1, [tiedAgentPos[0], ropePosList[-1]])
            else:
                tiedPosList.insert(0, [tiedAgentPos[0], ropePosList[0]])
                tiedPosList.insert(-1, [tiedAgentPos[1], ropePosList[-1]])
            [pg.draw.lines(self.screen, self.ropeColor, False, tiedPos, int(self.ropeWidth * ropeWidthRatio)) for tiedPos in tiedPosList]

        for agentIndex in range(self.numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        pg.time.wait(1)

        return self.screen

class ChaseTrialWithTraj:
    def __init__(self, fps, colorSpace, drawState, saveImage):
        self.fps = fps
        self.colorSpace = colorSpace
        self.drawState = drawState
        self.saveImage = saveImage

    def __call__(self, numOfAgents, trajectoryData, imagePath):
        fpsClock = pg.time.Clock()

        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(self.fps)
            screen = self.drawState(numOfAgents, state, self.colorSpace)

            if self.saveImage == True:
                pg.image.save(screen, imagePath + '/' + format(timeStep, '04') + ".png")

        return

class ChaseTrialWithRopeTraj:
    def __init__(self, fps, colorSpace, drawStateWithRope, saveImage, saveImageDir):
        self.fps = fps
        self.colorSpace = colorSpace
        self.drawStateWithRope = drawStateWithRope
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, trajectoryData, condition):
        fpsClock = pg.time.Clock()

        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(200)
            screen = self.drawStateWithRope(state, condition, self.colorSpace)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")

        return

