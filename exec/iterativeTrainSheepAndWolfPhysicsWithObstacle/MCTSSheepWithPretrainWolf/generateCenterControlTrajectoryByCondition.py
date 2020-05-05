import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import itertools as it
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue,ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal, TransitionFunction, ResetUniform
import mujoco_py as mujoco
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, RandomPolicy,HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from src.episode import chooseGreedyAction,SampleTrajectory




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

        xPos = np.concatenate(self.simulation.data.site_xpos[:self.numAgent, :self.numJointEachSite])

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentXPos = lambda agentIndex: xPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate(
            [agentQPos(agentIndex), agentXPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

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
                self.render(state,runningStep)
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)

            state = nextState
        return trajectory



def main():
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])

    # parametersForTrajectoryPath['sampleOneStepPerTraj']=1 #0
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    trainSteps = int(parametersForTrajectoryPath['trainSteps'])
    depth=int(parametersForTrajectoryPath['depth'])
    dataSize=int(parametersForTrajectoryPath['dataSize'])



    # parametersForTrajectoryPath = {}
    # depth = 5
    # dataSize = 5000
    # trainSteps = 50000
    # startSampleIndex = 0
    # endSampleIndex = 100


    killzoneRadius = 2
    numSimulations = 200
    maxRunningSteps = 50

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentPhysicsWithObstacleEnv4thWith(0,0)action','evaluateTrajectoriesStationarySheep')
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
        agentMaxSize=0
        wallList=[[0,2.5,0.8,1.95],[0,-2.5,0.8,1.95]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
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

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]

        numactionSpace=len(actionSpace)
        # neural network init
        numStateSpace = 12
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        generateWolvesModel = GenerateModel(numStateSpace, numactionSpace, regularizationFactor)
        initWolvesNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateSupervisedLearning', 'multiMCTSAgentPhysicsWithObstacleEnv4thWith(0,0)action', 'trainedResNNModels')

        wolfId = 1
        NNModelFixedParametersWolves = {'agentId': wolfId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'miniBatchSize':256,'learningRate':0.0001,}
        NNModelSaveExtension=' '
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParametersWolves)
        wolvesTrainedModelPath = getNNModelSavePath({'trainSteps':trainSteps,'depth':depth,'dataSize':dataSize})
        wolvesTrainedModel = restoreVariables(initWolvesNNModel, wolvesTrainedModelPath)
        wolfPolicy = ApproximatePolicy(wolvesTrainedModel, actionSpace)

        # randomSheepPolicy=RandomPolicy(actionSpace)
        # sheepPolicy=randomSheepPolicy
        sheepPolicy=lambda state:{(0,0):1}


        renderOn = False
        render=None
        if renderOn:

            from exec.evaluateNoPhysicsEnvWithRender import Render
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)
        chooseActionList = [chooseGreedyAction,chooseGreedyAction]
        sampleTrajectory = FixSampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)
        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        trajectories = [sampleTrajectory(policy,sampleIndex-startSampleIndex) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)

if __name__ == "__main__":
    main()