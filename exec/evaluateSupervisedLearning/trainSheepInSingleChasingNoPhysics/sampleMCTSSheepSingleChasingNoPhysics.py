import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, backup, InitializeChildren  # Expand, RollOut
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from src.episode import chooseGreedyAction  # SampleTrajectory
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset  # IsTerminal

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


def main():
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    agentId = int(parametersForTrajectoryPath['agentId'])
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])


    parametersForTrajectoryPath={}
    agentId = 0
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=100

    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # test
    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=5
    # test

    preyPowerRatio = 3
    predatorPowerRatio = 2
    killzoneRadius = 30

<<<<<<< HEAD
=======
    # imageFolderName = 'preyPowerRatio='+ str(preyPowerRatio) + '_predatorPowerRatio=' + str(predatorPowerRatio) +'_killzoneRadius=' + str(killzoneRadius)


>>>>>>> 29cab9810794495ed945f4c6b667f757388a814d
    numSimulations = 200
    maxRunningSteps = 150
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)

    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trajectoriesStillAction')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)



    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    # while True:
    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 2
        sheepId = 0
        wolfId = 1
        # wolf2Id = 2
        positionIndex = [0, 1]

        xBoundary = [0, 600]
        yBoundary = [0, 600]

<<<<<<< HEAD
        # prepare render

=======
        #prepare render
>>>>>>> 29cab9810794495ed945f4c6b667f757388a814d
        # from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender
        # import pygame as pg
        # renderOn = True
        # from pygame.color import THECOLORS
        # screenColor = THECOLORS['black']
        # circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        # circleSize = 10
        # screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        # saveImage = False
        # saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg',imageFolderName)
        # if not os.path.exists(saveImageDir):
        #     os.makedirs(saveImageDir)
        # render = Render(numOfAgent, positionIndex,
        #                 screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)
        # getPredator2Pos=GetAgentPosFromState(wolf2Id, positionIndex)
        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

        # isTerminal1 = env.IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)
        # isTerminal2 =env.IsTerminal(getPredator2Pos, getPreyPos, killzoneRadius)
        # isTerminal=lambda state:isTerminal1(state) or isTerminal1(state)

<<<<<<< HEAD
        divideDegree = 5
        isTerminal = IsTerminal(getPredatorPos, getPreyPos, killzoneRadius, divideDegree)
=======
        divideDegree=5
        isTerminal = IsTerminal(getPredatorPos, getPreyPos, killzoneRadius,divideDegree)
>>>>>>> 29cab9810794495ed945f4c6b667f757388a814d
        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]

        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
<<<<<<< HEAD
        sheepActionSpace.append((0, 0))
=======
        sheepActionSpace.append((0,0))
        numActionSpace = len(actionSpace)


        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2

        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        sheepActionSpace.append((0,0))
>>>>>>> 29cab9810794495ed945f4c6b667f757388a814d

        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        numActionSpace = len(sheepActionSpace)

        wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy
        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolf1Policy(state))])

        # reward function
        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            sheepActionSpace, sheepTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return sheepActionSpace[np.random.choice(range(numActionSpace))]
        # rollout
        rolloutHeuristicWeight = 0
        rolloutHeuristic = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getPredatorPos, getPreyPos)
        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                          rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                           rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolf1Policy(state)]

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:', len(trajectories[0]))
        print('timeTaken:', finshedTime)


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0

        if leafNode.is_root:
            lastState = currentState
        else:
            lastState = list(leafNode.parent.id.values())[0]

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            totalRewardForRollout += self.rewardFunction(lastState, currentState, action)
            if self.isTerminal(lastState, currentState):
                break
            nextState = self.transitionFunction(currentState, action)
            lastState = currentState
            currentState = nextState

        heuristicReward = 0
        if not self.isTerminal(lastState, currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout


class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance, divideDegree):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance
        self.divideDegree = divideDegree

    def __call__(self, lastState, currentState):
        terminal = False

        getPositionList = lambda getPos, lastState, currentState: np.linspace(getPos(lastState), getPos(currentState), self.divideDegree, endpoint=True)

        getL2Normdistance = lambda preyPosition, predatorPosition: np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)

        preyPositionList = getPositionList(self.getPreyPos, lastState, currentState)
        predatorPositionList = getPositionList(self.getPredatorPos, lastState, currentState)

        L2NormdistanceList = [getL2Normdistance(preyPosition, predatorPosition) for (preyPosition, predatorPosition) in zip(preyPositionList, predatorPositionList)]

        if np.any(np.array(L2NormdistanceList) <= self.minDistance):
            terminal = True
        return terminal


class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if leafNode.is_root:
            lastState = currentState
        else:
            lastState = list(leafNode.parent.id.values())[0]
        if not self.isTerminal(lastState, currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state, state):
            state = self.reset()
            print(L2NormdistanceList)
            print(np.array(L2NormdistanceList) <= self.minDistance)

        trajectory = []
        lastState = state
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(lastState, state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            lastState = state
            state = nextState

        return trajectory


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

        while self.isTerminal(state, state):
            state = self.reset()

        trajectory = []
        lastState = state
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(lastState, state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
<<<<<<< HEAD
                self.render(state, runningStep)
=======
                self.render(state)
>>>>>>> 29cab9810794495ed945f4c6b667f757388a814d
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            lastState = state
            state = nextState

        return trajectory


class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, lastState, currentState, action):
        reward = self.aliveBonus
        if self.isTerminal(lastState, currentState):
            reward += self.deathPenalty

        return reward


if __name__ == "__main__":
    main()
