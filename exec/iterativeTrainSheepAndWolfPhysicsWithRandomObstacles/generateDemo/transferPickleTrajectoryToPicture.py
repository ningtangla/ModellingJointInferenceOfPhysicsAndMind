

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import itertools as it
import pygame as pg
from pygame.color import THECOLORS

from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue,ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal, TransitionFunction, ResetUniform
import mujoco_py as mujoco
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, RandomPolicy,HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from src.episode import chooseGreedyAction,SampleTrajectory

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle,LoadTrajectories,loadFromPickle






class TransferWallToRescalePosForDraw:
    def __init__(self,rawXRange,rawYRange,scaledXRange,scaledYRange):
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange
        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)
        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (self.rawYMax-rawY) * yScale + self.scaledYMin
        self.rescaleWall=lambda wallForDraw :[adjustX(wallForDraw[0]),adjustY(wallForDraw[1]),wallForDraw[2]*xScale,wallForDraw[3]*yScale]
        self.tranferWallForDraw=lambda wall:[wall[0]-wall[2],wall[1]+wall[3],2*wall[2],2*wall[3]]
    def __call__(self,wallList):

        wallForDarwList=[self.tranferWallForDraw(wall) for wall in wallList]
        allObstaclePos=[ self.rescaleWall(wallForDraw) for wallForDraw in wallForDarwList]
        return allObstaclePos




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
        adjustY = lambda rawY: (self.rawYMax-rawY) * yScale + self.scaledYMin

        adjustState = lambda state: [adjustX(state[self.xIndex]), adjustY(state[self.yIndex])]

        newState = [adjustState(agentState) for agentState in originalState]

        return newState


def main():


    # check file exists or not
    dirName = os.path.dirname(__file__)

    dataFolderName=os.path.join(dirName,'..','..', '..', 'data')

    mainSaveImageDir = os.path.join(dataFolderName,'obstacaleDemoImg','test')
    if not os.path.exists(mainSaveImageDir):
        os.makedirs(mainSaveImageDir)

    trajectorySaveExtension = '.pickle'
#######
    #MCTS Trajectories
    # trajectoryDirectory = os.path.join(dataFolderName, 'multiAgentTrain', 'MCTSRandomObstacle',  'trajectories')
    # numSimulations=150
    # maxRolloutSteps=30
    # agentId=1
    # maxRunningSteps = 30
    # killzoneRadius = 2
    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius,'maxRolloutSteps':maxRolloutSteps,'agentId':agentId}



######
    # #Iterative NNGuideMCTS Trajectories
    # numSimulations=200
    # maxRunningSteps = 30
    # killzoneRadius = 2
    # depth=4
    # learningRate=0.001
    # trajectoriesSaveDirectory=os.path.join(dataFolderName, 'multiAgentTrain', 'multiMCTSAgentFixObstacle','trajectories')
    # fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'depth':depth,'learningRate':learningRate}
    # generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    # loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)
    # restoredIterationIndexRange = range(2000,2003)
    # restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})


######
    #supervise ResNN NN
    # killzoneRadius = 2
    # numSimulations = 150
    # maxRunningSteps = 30
    # agentId=1
    # depth=4
    # learningRate=1e-5
    # miniBatchSize=256
    # trainSteps=180000
    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'depth':depth,'learningRate':learningRate,'miniBatchSize':miniBatchSize,'trainSteps':trainSteps,'agentId':agentId}
    # dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'MCTSRandomObstacle')
    # trajectoryDirectory = os.path.join(dataFolderName,'evaluationTrajectoriesNNWithObstacle')
#########
    # #iterativeTrain add wall
    dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'originVersionAddObstacle')
    trajectoryDirectory = os.path.join(dataFolderName,'evaluateTrajectories')
    trainMaxRunningSteps = 30
    trainNumSimulations = 200
    killzoneRadius=2
    selfIteration=5000
    otherIteration=5000
    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius,'selfIteration':selfIteration,'otherIteration':otherIteration}


########
    





########
    generateTrajectoryLoadPath = GetSavePath(trajectoryDirectory, trajectorySaveExtension, trajectoryFixedParameters)


    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForDraw = LoadTrajectories(generateTrajectoryLoadPath, loadFromPickle, fuzzySearchParameterNames)

    restoredTrajectories = loadTrajectoriesForDraw({})





    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = pg.display.set_mode([screenWidth, screenHeight])

    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]


    leaveEdgeSpace = 200
    lineWidth = 3
    xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
    yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]

    positionIndex = [2, 3]

    sheepId = 0
    wolfId = 1
    angetIdList=[sheepId,wolfId]
    wallIdlist=[2,3]
    getAgentState=lambda state,i:state[i]
    getObstaclePos=lambda state,i:state[i]
    parseState=ParseState(angetIdList,wallIdlist,getAgentState,getObstaclePos)



    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [210, 590]
    scaledYRange = [210, 590]
    scaleState = ScaleState(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)
    scaleWall=TransferWallToRescalePosForDraw(rawXRange,rawYRange,scaledXRange,scaledYRange)

    screenColor = THECOLORS['black']
    lineColor = THECOLORS['white']
    drawBackGround = DrawBackgroundWithObstacles(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

    circleSizeList=[8,11]
    drawState = DrawState(screen, circleSizeList,circleColorList, positionIndex)

    saveImage = False

    numOfAgent=2
    render = RenderInObstacle(numOfAgent,screen, circleColorList, saveImage, parseState,scaleWall,drawBackGround,scaleState,drawState)

    stateId=0
    for n,trajectory in enumerate(restoredTrajectories):

        saveImageDir=os.path.join(mainSaveImageDir,str(n))

        for step,state in enumerate(trajectory) :
            render(state[0],step,saveImageDir)





class RenderInObstacle():
    def __init__(self, numOfAgent, screen, circleColorList,saveImage,parseState,scaleWall,drawBackGround,scaleState,drawState):
        self.numOfAgent = numOfAgent
        self.screen = screen
        self.circleColorList = circleColorList
        self.saveImage  = saveImage
        self.parseState=parseState
        self.scaleWall=scaleWall
        self.drawBackGround=drawBackGround
        self.scaleState=scaleState
        self.drawState=drawState
    def __call__(self, state, timeStep,saveImageDir):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

            agentState,wallList=self.parseState(state)

            allObstaclePos = self.scaleWall(wallList)
            self.drawBackGround(allObstaclePos)

            rescaleState=self.scaleState(agentState)
            # print(state,rescaleState)
            screen = self.drawState(self.numOfAgent,rescaleState)
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(saveImageDir):
                    os.makedirs(saveImageDir)
                pg.image.save(self.screen, saveImageDir + '/' + format(timeStep, '05') + ".png")

class TransferWallToRescalePosForDraw:
    def __init__(self,rawXRange,rawYRange,scaledXRange,scaledYRange):
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange
        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)
        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (self.rawYMax-rawY) * yScale + self.scaledYMin
        self.rescaleWall=lambda wallForDraw :[adjustX(wallForDraw[0]),adjustY(wallForDraw[1]),wallForDraw[2]*xScale,wallForDraw[3]*yScale]
        self.tranferWallForDraw=lambda wall:[wall[0]-wall[2],wall[1]+wall[3],2*wall[2],2*wall[3]]
    def __call__(self,wallList):

        wallForDarwList=[self.tranferWallForDraw(wall) for wall in wallList]
        allObstaclePos=[ self.rescaleWall(wallForDraw) for wallForDraw in wallForDarwList]
        return allObstaclePos
class DrawState:
    def __init__(self, screen, circleSizeList,circleColorList, positionIndex):
        self.screen = screen
        self.circleSizeList = circleSizeList
        self.xIndex, self.yIndex = positionIndex
        self.circleColorList=circleColorList
    def __call__(self, numOfAgent, state):


        for agentIndex in range(numOfAgent):
            agentPos =[np.int(pos) for pos in state[agentIndex]]
            agentColor = self.circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSizeList[agentIndex])
        pg.display.flip()
        return self.screen
class DrawBackgroundWithObstacles:
    def __init__(self, screen, screenColor, xBoundary, yBoundary,  lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self,allObstaclePos):
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
         allObstaclePos]

        return
class ParseState():
    def __init__(self, angetIdList,wallIdlist,getAgentState,getObstaclePos):
        self.angetIdList = angetIdList
        self.wallIdlist = wallIdlist
        self.getAgentState = getAgentState
        self.getObstaclePos  = getObstaclePos

    def __call__(self, state):

        agentState=[self.getAgentState(state,i) for i in self.angetIdList]
        wallList=[self.getObstaclePos(state,i) for i in self.wallIdlist]

        return agentState,wallList

if __name__ == "__main__":
    main()
