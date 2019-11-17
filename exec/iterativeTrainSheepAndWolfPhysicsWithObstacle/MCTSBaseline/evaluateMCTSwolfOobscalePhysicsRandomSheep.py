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
import itertools as it
import mujoco_py as mujoco
import pathos.multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,HeuristicDistanceToTarget

from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.constrainedChasingEscapingEnv.policies import RandomPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory,AccumulateRewards,ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist,Expand,RollOut,establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.evaluationFunctions import ComputeStatistics

class IsResetOnTerminal:
    def __init__(self,killZoneRaius):
        self.killZoneRaius=killZoneRaius
    def __call__(self,qPos):
        pointList=qPos.reshape(-1,2)
        isTerminalList=[np.linalg.norm((pos0 - pos1), ord=2)<self.killZoneRaius  for pos0,pos1 in it.combinations(pointList,2)]
        return np.any(isTerminalList)

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
        while self.checkAngentStackInWall(qPos) or self.isResetOnTerminal(qPos):
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)
        return [qPos,qVel]
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

        while self.isTerminal(state):
            state = self.reset(trailIndex)

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


def generateOneCondition(parameterOneCondition):
    print(parameterOneCondition)
    numSimulations = int(parameterOneCondition['numSimulations'])
    maxRolloutSteps=int(parameterOneCondition['maxRolloutSteps'])
    numTrials=20#20
    maxRunningSteps = 30
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'numTrials': numTrials,'maxRolloutSteps':maxRolloutSteps}

    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiMCTSAgentPhysicsWithObstacle','evaluateMCTSSimulation', 'trajectories')
    trajectorySaveExtension = '.pickle'

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parameterOneCondition)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath=os.path.join(dirName,'..','twoAgentsTwoObstacles2.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        agentMaxSize=0
        # wallList=[[0,3,0.5,2.8],[0,-3,0.5,2.8]]
        wallList=[[0,2,0.5,1.75],[0,-2,0.5,1.75]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7

        np.random.seed(1447)
        isResetOnTerminal=IsResetOnTerminal(killzoneRadius)
        samplePositionInObstaclesEnv=SamplePositionInObstaclesEnv(physicsSimulation,qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,checkAngentStackInWall,isResetOnTerminal)
        initPositionList = [samplePositionInObstaclesEnv() for i in range(numTrials)]
        print(numSimulations,initPositionList)



        reset = FixResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,initPositionList)

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

        numSimulationFrames=20
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


        randomSheepPolicy=RandomPolicy(actionSpace)
        sheepPolicy=randomSheepPolicy




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
        rolloutHeuristicWeight = 0.0
        rolloutHeuristic = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfXPos, getSheepXPos)


        # maxRolloutSteps = 20
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit,rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]

        chooseActionList = [chooseGreedyAction,chooseGreedyAction]

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


        sampleTrajectory = FixSampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        trajectories = [sampleTrajectory(policy,sampleIndex) for sampleIndex in range(numTrials)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [50, 100, 200, 400]
    manipulatedVariables['maxRolloutSteps']  = [10,20,30]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(generateOneCondition, parametersAllCondtion)

    # load data
    dirName = os.path.dirname(__file__)
    trajectoryDirectory=os.path.join(dirName, '..', '..', '..', 'data', 'multiMCTSAgentPhysicsWithObstacle','evaluateMCTSSimulation', 'trajectories')
    trajectoryExtension = '.pickle'
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)


    numTrials = 20
    killzoneRadius = 2
    maxRunningSteps = 30

    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius, 'numTrials': numTrials}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))
    sheepId = 0
    wolfId = 1

    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    playIsTerminal = IsTerminal(killzoneRadius,getWolfXPos, getSheepXPos)

    playAliveBonus = -1 / maxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)

    # compute statistics on the trajectories
    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)
    # plot the results
    fig = plt.figure()
    numRows = 1
    numColumns = len(manipulatedVariables['numSimulations'])
    plotCounter = 1


    for numSimulation,group in statisticsDf.groupby('numSimulations'):
        group.index=group.index.droplevel('numSimulations')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_title('numSimulations: {}'.format(numSimulation))
        axForDraw.set_ylim(-1, 1)
        group.plot(ax=axForDraw, y='mean', yerr='std', marker='o', logx=False)
        plt.ylabel('Accumulated rewards')
        plt.xlim(0)
        plotCounter += 1



    plt.suptitle('Evaulate MCTS wolf with random sheep in Obstacles Physics')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
