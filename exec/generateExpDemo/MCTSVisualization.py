import os
import numpy as np
import pandas as pd
import pygame as pg

class ScalePos:
    def __init__(self, positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange):
        self.xIndex, self.yIndex = positionIndex
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange

        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange

    def __call__(self, state):
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)

        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (rawY - self.rawYMin) * yScale + self.scaledYMin

        adjustPosPair = lambda pair: [adjustX(pair[self.xIndex]), adjustY(pair[self.yIndex])]
        agentCount = len(state)

        adjustPos = lambda state: [adjustPosPair(state[agentIndex]) for agentIndex in range(agentCount)]
        adjustedPoses = adjustPos(state) 

        return adjustedPoses

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

class MctsRender():
    def __init__(self, numAgent, MCTSAgentId, tiedAgentId, screen, surfaceWidth, surfaceHeight, screenColor, circleColorList, circleSize, saveImage, saveImageFile, drawStateWithRope, scaleState):
        self.numAgent = numAgent
        self.MCTSAgentId = MCTSAgentId
        self.tiedAgentId = tiedAgentId
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
        self.drawStateWithRope = drawStateWithRope
	self.scalePos = scalePos

    def __call__(self, currNode, nextNode, roots, backgroundScreen):

        parentNumVisit = currNode.num_visited
        parentValueToTal = currNode.sum_value
        originalState = list(currNode.id.values())[0] 
        poses = self.scalePos(originalState)
         
        childNumVisit = nextNode.num_visited
        childValueToTal = nextNode.sum_value
        originalNextState = list(nextNode.id.values())[0]
        nextPoses = self.scalePos(originalNextState)

        lineWidth = nextNode.num_visited + 1 
        if len(roots) > 0 and nextNode.depth == 1:
            nodeIndex = currNode.children.index(nextNode)
            grandchildren_visit = np.sum([[child.num_visited for child in anytree.findall(root, lambda node: node.depth == 1)] for root in roots], axis=0)
            lineWidth = lineWidth + grandchildren_visit[nodeIndex] 

        surfaceToDraw = pg.Surface((self.surfaceWidth, self.surfaceHeight))
        surfaceToDraw.fill(self.screenColor)
        if backgroundScreen == None:
            backgroundScreen = self.drawStateWithRope(state, self.tiedAgentId, self.circleColorList)
            if self.saveImage==True:
                currentDir = os.getcwd()
                parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                saveImageDir=os.path.join(parentDir+'/src/data/'+self.saveImageFile)
                filenameList = os.listdir(saveImageDir)
                pg.image.save(backgroundScreen,saveImageDir+'/'+str(len(filenameList))+'.png')
         
        surfaceToDraw.set_alpha(180)
        surfaceToDraw.blit(backgroundScreen, (0,0))
        self.screen.blit(surfaceToDraw, (0, 0)) 
    
        pg.display.flip()
        pg.time.wait(1)
        
        
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
                
            for i in range(self.numAgent):
                oneAgentPos = np.array(poses[i])
                oneAgentNextPosition = np.array(nextPoses[i])
                if i == self.MCTSAgentId: 
                    pg.draw.line(surfaceToDraw, np.ones(3) * 240, [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], lineWidth)
                    pg.draw.circle(surfaceToDraw, self.circleColorList[i], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], self.circleSize)
             
            self.screen.blit(surfaceToDraw, (0, 0)) 
            pg.display.flip()
            pg.time.wait(1)
            
            if self.saveImage==True:
                currentDir = os.getcwd()
                parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                saveImageDir=parentDir+'/src/data/'+self.saveImageFile
                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
        return self.screen

