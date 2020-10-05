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

from src.episode import chooseGreedyAction,SampleTrajectory,SampleAction

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle

class CheckAngentStackInWall:
    def __init__(self, wallList,agentMaxSize):
        self.wallList=wallList
        self.agentMaxSize=agentMaxSize
    def __call__(self,qPosList):
        wallCenterList=np.array([wall[:2] for wall in self.wallList])
        wallExpandHalfDiagonalList=np.array([np.add(wall[2:],self.agentMaxSize) for wall in self.wallList])
        posList=qPosList.reshape(-1,2)
        isOverlapList=[np.all(np.abs(np.add(pos,-center))<diag)  for (center,diag) in zip (wallCenterList,wallExpandHalfDiagonalList) for pos in posList]
        return np.any(isOverlapList)
class SamplePositionInObstaclesEnv:
    def __init__(self,simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,checkAngentStackInWall,isResetOnTerminal):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = numAgent
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.isResetOnTerminal=isResetOnTerminal
        self.checkAngentStackInWall=checkAngentStackInWall
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        while self.checkAngentStackInWall(qPos) or self.isResetOnTerminal(qPos) :
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)
        return [qPos,qVel]
class IsResetOnTerminal:
    def __init__(self,killZoneRaius):
        self.killZoneRaius=killZoneRaius
    def __call__(self,qPos):
        pointList=qPos.reshape(-1,2)
        isTerminalList=[np.linalg.norm((pos0 - pos1), ord=2)<self.killZoneRaius  for pos0,pos1 in it.combinations(pointList,2)]
        return np.any(isTerminalList)
class FixResetUniformInEnvWithObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,resetList):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.resetList=resetList
    def __call__(self,trailIndex):

        qPos,qVel =self.resetList[trailIndex]

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        # xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])
        xPos=qPos
        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])
        print(startState)
        return startState

class FixSampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn
        self.runningStep=0
    def __call__(self, policy,trailIndex):
        state = self.reset(trailIndex)

        # while self.isTerminal(state):
        #     state = self.reset(trailIndex)
        #     print('isTerminal')

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state,self.runningStep)
            self.runningStep=self.runningStep+1
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            # print(nextState)
            state = nextState
        return trajectory



def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # trainSteps = int(parametersForTrajectoryPath['trainSteps'])
    # depth=int(parametersForTrajectoryPath['depth'])
    # dataSize=int(parametersForTrajectoryPath['dataSize'])

    parametersForTrajectoryPath = {}


    startSampleIndex = 0
    endSampleIndex = 100

    sheepIteration=2000
    wolfIteration=2000

    killzoneRadius = 2
    numSimulations = 200
    maxRunningSteps = 30

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetObstacle','demoTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath=os.path.join(dirName,'..','twoAgentsTwoObstacles4.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        agentMaxSize=0.6
        wallList=[[0,2.5,0.8,1.95],[0,-2.5,0.8,1.95]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 0
        qPosInitNoise = 9.7
        numTrials=endSampleIndex-startSampleIndex
        np.random.seed(startSampleIndex*endSampleIndex)
        isResetOnTerminal=IsResetOnTerminal(killzoneRadius)
        samplePositionInObstaclesEnv=SamplePositionInObstaclesEnv(physicsSimulation,qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,checkAngentStackInWall,isResetOnTerminal)
        initPositionList = [samplePositionInObstaclesEnv() for i in range(numTrials)]
        reset = FixResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,initPositionList)

        agentIds = list(range(numAgents))
        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

        numactionSpace=len(actionSpace)
        # neural network init
        numStateSpace = 12
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        resBlockSize = 2
        dropoutRate = 0.0
        depth =9
        initializationMethod = 'uniform'
        generateModel = GenerateModel(numStateSpace, numactionSpace, regularizationFactor)

        initWolfNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetObstacle', 'NNModel')
        NNModelSaveExtension=' '
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        wolfModelPath = getNNModelSavePath({'iterationIndex': wolfIteration, 'agentId': wolfId})
        wolvesTrainedModel = restoreVariables(initWolfNNModel, wolfModelPath)
        wolfPolicy = ApproximatePolicy(wolvesTrainedModel, actionSpace)

        initSheepNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = getNNModelSavePath({'iterationIndex': sheepIteration, 'agentId': sheepId})
        restoreSheepNNModel=restoreVariables(initSheepNNModel, sheepModelPath)
        sheepPolicy = ApproximatePolicy(restoreSheepNNModel, actionSpace)

        renderOn = True
        render=None
        if renderOn:
            from visualize.continuousVisualization import DrawBackgroundWithObstacles
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
            rawXRange = [-10, 10]
            rawYRange = [-10, 10]
            scaledXRange = [210, 590]
            scaledYRange = [210, 590]
            scaleState = ScaleState(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)
            transferWallToRescalePosForDraw=TransferWallToRescalePosForDraw(rawXRange,rawYRange,scaledXRange,scaledYRange)

            allObstaclePos = transferWallToRescalePosForDraw(wallList)

            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']
            drawBackground = DrawBackgroundWithObstacles(screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth)

            circleSizeList=[8,11]
            drawState = DrawState(screen, circleSizeList,circleColorList, positionIndex,drawBackground)

            saveImage = True
            demoDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetObstacle', 'NNdemo',str(sheepIteration))

            if not os.path.exists(demoDirectory):
                os.makedirs(demoDirectory)
            numOfAgent=2
            render = RenderInObstacle(numOfAgent, positionIndex,screen, circleColorList, saveImage, demoDirectory,scaleState,drawState)

        beta=1
        sampleAction=SampleAction(beta)
        chooseActionList = [chooseGreedyAction,chooseGreedyAction]
        chooseActionList=[sampleAction,sampleAction]
        playMaxRunningSteps=100
        sampleTrajectory = FixSampleTrajectoryWithRender(playMaxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)
        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        trajectories = [sampleTrajectory(policy,sampleIndex-startSampleIndex) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)

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

class RenderInObstacle():
    def __init__(self, numOfAgent, posIndex, screen, circleColorList,saveImage, saveImageDir,scaleState,drawState):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.circleColorList = circleColorList
        self.saveImage  = saveImage
        self.saveImageDir = saveImageDir
        self.scaleState=scaleState
        self.drawState=drawState
    def __call__(self, state, timeStep):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

            rescaleState=self.scaleState(state)
            # print(state,rescaleState)
            screen = self.drawState(self.numOfAgent,rescaleState)
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")
class DrawState:
    def __init__(self, screen, circleSizeList,circleColorList, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSizeList = circleSizeList
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround
        self.circleColorList=circleColorList
    def __call__(self, numOfAgent, state):

        self.drawBackGround()

        for agentIndex in range(numOfAgent):
            agentPos =[np.int(pos) for pos in state[agentIndex]]
            agentColor = self.circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSizeList[agentIndex])
        pg.display.flip()
        return self.screen

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
if __name__ == "__main__":
    main()
