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

from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.constrainedChasingEscapingEnv.policies import RandomPolicy
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

class ResetUniformInEnvWithObstacles:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise,checkAngentStackInWall):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.checkAngentStackInWall=checkAngentStackInWall
    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)


        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)

        while self.checkAngentStackInWall(qPos):
            qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        qVel = self.qVelInit + np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel)

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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentPhysicsWithObstacle', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 200
    killzoneRadius = 2
    maxRolloutSteps = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'maxRolloutSteps':maxRolloutSteps}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=12
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath=os.path.join(dirName,'..','twoAgentsTwoObstacles3.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        # MDP function
        agentMaxSize=0.6
        wallList=[[0,2,0.5,1.45],[0,-2,0.5,1.45]]
        checkAngentStackInWall=CheckAngentStackInWall(wallList,agentMaxSize)

        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7

        reset = ResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,checkAngentStackInWall)

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


        randomSheepPolicy=RandomPolicy(actionSpace)
        sheepPolicy=randomSheepPolicy



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
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)


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



        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit,rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
