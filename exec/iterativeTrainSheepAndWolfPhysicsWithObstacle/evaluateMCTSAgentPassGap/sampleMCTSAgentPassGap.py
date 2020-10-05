
import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
import mujoco_py as mujoco
import xmltodict
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.constrainedChasingEscapingEnv.policies import RandomPolicy,stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist,Expand,RollOut,establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel





def transferNumberListToStr(numList):
    strList=[str(num) for num in numList]
    return ' '.join(strList)
def changeWallProperty(envDict,wallPropertyDict):
    for number,propertyDict in wallPropertyDict.items():
        for name,value in propertyDict.items():
            envDict['mujoco']['worldbody']['body'][number]['geom'][name]=value

    return envDict
def main():

    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, 'twoAgentsTwoObstacles2.xml')
    xml_dict=parse_file(physicsDynamicsPath)

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    xml_doc_dict = xmltodict.parse(xml_string.strip())

    wallPropertyDict={}
    wall1Id=5
    wall2Id=6
    gapLenth=1.6
    wall1Pos=[0,(9.95+gapLenth/2)/2,-0.2]
    wall1Size=[0.9,(9.95+gapLenth/2)/2-gapLenth/2,1.5]
    wall2Pos=[0,-(9.95+gapLenth/2)/2,-0.2]
    wall2Size=[0.9,(9.95+gapLenth/2)/2-gapLenth/2,1.5]

    wallPropertyDict[wall1Id]={'@pos':transferNumberListToStr(wall1Pos),'@size':transferNumberListToStr(wall1Size)}
    wallPropertyDict[wall2Id]={'@pos':transferNumberListToStr(wall2Pos),'@size':transferNumberListToStr(wall2Size)}

    xml_doc_dict=changeWallProperty(xml_doc_dict,wallPropertyDict)
    xml=xmltodict.unparse(xml_doc_dict)
    physicsModel = mujoco.load_model_from_xml(xml)
    physicsSimulation = mujoco.MjSim(physicsModel)
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

class ResetUniformInEnvWithObstaclesOnHalfSideWithFixSheep:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,sheepPos,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.checkAngentStackInWall=checkAngentStackInWall
        self.sheepPos=np.array(sheepPos)
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)


        qPos = self.qPosInit +np.append(self.sheepPos, np.random.uniform(low=0, high=self.qPosInitNoise, size=numQPos-2))

        while self.checkAngentStackInWall(qPos):
             qPos = self.qPosInit +np.append(self.sheepPos, np.random.uniform(low=0, high=self.qPosInitNoise, size=numQPos-2))
        qVel = self.qVelInit +np.append( np.array([0,0]),np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel-2))

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
class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            # print(state)
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
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','evaluateMCTSAgentPassGapSheep',  'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    numSimulations = 50
    killzoneRadius = 2
    maxRolloutSteps = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'maxRolloutSteps':maxRolloutSteps}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    np.random.seed(startSampleIndex*endSampleIndex)
    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=2
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # parametersForTrajectoryPath['gapLength']=1.6
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, 'twoAgentsOneDoor.xml')

        with open(physicsDynamicsPath) as f:
            xml_string = f.read()
        xml_doc_dict = xmltodict.parse(xml_string.strip())

        wallPropertyDict={}
        wall1Id=5
        wall2Id=6
        gapLength=float(parametersForTrajectoryPath['gapLength'])
        wall1Pos=[0,(9.95+gapLength/2)/2,-0.2]
        wall1Size=[0.9,(9.95+gapLength/2)/2-gapLength/2,1.5]
        wall2Pos=[0,-(9.95+gapLength/2)/2,-0.2]
        wall2Size=[0.9,(9.95+gapLength/2)/2-gapLength/2,1.5]

        wallPropertyDict[wall1Id]={'@pos':transferNumberListToStr(wall1Pos),'@size':transferNumberListToStr(wall1Size)}
        wallPropertyDict[wall2Id]={'@pos':transferNumberListToStr(wall2Pos),'@size':transferNumberListToStr(wall2Size)}

        xml_doc_dict=changeWallProperty(xml_doc_dict,wallPropertyDict)
        xml=xmltodict.unparse(xml_doc_dict)
        physicsModel = mujoco.load_model_from_xml(xml)
        physicsSimulation = mujoco.MjSim(physicsModel)

        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        agentMaxSize=0.6
        wallList=[[0,5,0.9,5],[0,-5,0.9,5]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7
        sheepPos=[-3,0]
        reset = ResetUniformInEnvWithObstaclesOnHalfSideWithFixSheep(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,sheepPos,checkAngentStackInWall)

        agentIds = list(range(numAgents))
        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus

        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]


        # randomSheepPolicy=RandomPolicy(actionSpace)
        sheepPolicy=stationaryAgentPolicy




        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    # load chase nn policy

        def wolvesTransit(state, action):
             return transit(
            state, [chooseGreedyAction(sheepPolicy(state)),action])

        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            actionSpace, wolvesTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        numWolfActionSpace=len(actionSpace)
        def rolloutPolicy(
            state): return actionSpace[np.random.choice(range(numWolfActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0.1
        rolloutHeuristic = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfXPos, getSheepXPos)



        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit,rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]


        renderOn = False
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

            circleSizeList=[8,8]
            drawState = DrawState(screen, circleSizeList,circleColorList, positionIndex,drawBackground)

            saveImage = False
            demoDirectory = os.path.join(dirName, '..', '..', '..', 'data')

            if not os.path.exists(demoDirectory):
                os.makedirs(demoDirectory)
            numOfAgent=2
            render = RenderInObstacle(numOfAgent, positionIndex,screen, circleColorList, saveImage, demoDirectory,scaleState,drawState)
        chooseActionList = [chooseGreedyAction,chooseGreedyAction]
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
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

if __name__ == '__main__':
    main()