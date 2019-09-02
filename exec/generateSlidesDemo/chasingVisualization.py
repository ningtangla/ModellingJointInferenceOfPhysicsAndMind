import pygame as pg
import os
import numpy as np


class InitializeScreen:
    def __init__(self, screenWidth, screenHeight, fullScreen):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.fullScreen = fullScreen

    def __call__(self):
        pg.init()
        if self.fullScreen:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight), pg.FULLSCREEN)
        else:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight))
        return screen


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


class DrawState:
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround

    def __call__(self, state, circleColorList):
        self.drawBackGround()
        for agentIndex in range(self.numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        return self.screen


class DrawStateWithRope():
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, ropePartIndex, ropeColor, ropeWidth, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.ropePartIndex = ropePartIndex
        self.ropeColor = ropeColor
        self.ropeWidth = ropeWidth
        self.drawBackGround = drawBackGround

    def __call__(self, state, tiedAgentId, circleColorList):
        self.drawBackGround()
        if tiedAgentId:
            tiedAgentPos = [[np.int(state[agentId][self.xIndex]), np.int(state[agentId][self.yIndex])] for agentId in tiedAgentId]
            ropePosList = [[np.int(state[ropeId][self.xIndex]), np.int(state[ropeId][self.yIndex])] for ropeId in self.ropePartIndex]

            tiedPosList = [[ropePosList[i], ropePosList[i + 1]] for i in range(0, len(ropePosList) - 1)]
            tiedPosList.insert(0, [tiedAgentPos[0], ropePosList[0]])
            tiedPosList.insert(-1, [tiedAgentPos[1], ropePosList[-1]])

            [pg.draw.lines(self.screen, self.ropeColor, False, tiedPos, self.ropeWidth) for tiedPos in tiedPosList]

        for agentIndex in range(self.numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        pg.time.wait(10)

        return self.screen


class ChaseTrialWithTraj:
    def __init__(self, fps, colorSpace,
                 drawState, saveImage, saveImageDir):

        self.fps = fps
        self.colorSpace = colorSpace
        self.drawState = drawState
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, trajectoryData):
        fpsClock = pg.time.Clock()

        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(200)
            screen = self.drawState(state, self.colorSpace)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")

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
