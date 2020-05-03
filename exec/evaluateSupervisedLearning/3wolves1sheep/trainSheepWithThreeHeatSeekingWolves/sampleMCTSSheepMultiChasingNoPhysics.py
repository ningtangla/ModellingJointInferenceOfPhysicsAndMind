import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import numpy as np
import pickle
import random
import json
import time

from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset, TransitWithInterpolateState
from src.episode import SampleTrajectory, Render, SampleTrajectoryWithRender, chooseGreedyAction, SampleAction
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


def main():
    DEBUG = 0
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 10
        agentId = 0
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        agentId = int(parametersForTrajectoryPath['agentId'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    numSimulations = 110
    maxRunningSteps = 50
    killzoneRadius = 50
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '3wolves1sheep', 'trainSheepInMultiChasingNoPhysicsThreeWolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    # while True:
    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 4
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        wolfThreeId = 3
        positionIndex = [0, 1]

        xBoundary = [0, 600]
        yBoundary = [0, 600]

        getSheepPos = GetAgentPosFromState(sheepId, positionIndex)
        getWolfOnePos = GetAgentPosFromState(wolfOneId, positionIndex)
        getWolfTwoPos = GetAgentPosFromState(wolfTwoId, positionIndex)
        getWolfThreePos = GetAgentPosFromState(wolfThreeId, positionIndex)

        isTerminalOne = env.IsTerminal(getWolfOnePos, getSheepPos, killzoneRadius)
        isTerminalTwo = env.IsTerminal(getWolfTwoPos, getSheepPos, killzoneRadius)
        isTerminalThree = env.IsTerminal(getWolfThreePos, getSheepPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state) or isTerminalThree(state)

        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transite = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transitionFunction = TransitWithInterpolateState(numFramesToInterpolate, transite, isTerminal)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        numActionSpace = len(actionSpace)

        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getWolfOnePos, getSheepPos, computeAngleBetweenVectors)

        wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getWolfTwoPos, getSheepPos, computeAngleBetweenVectors)

        wolfThreePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getWolfThreePos, getSheepPos, computeAngleBetweenVectors)
        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy
        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)

        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseActionInMCTS(wolfOnePolicy(state)), chooseActionInMCTS(wolfTwoPolicy(state)), chooseActionInMCTS(wolfThreePolicy(state))])

        # reward function
        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        initializeChildren = InitializeChildren(
            sheepActionSpace, sheepTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return sheepActionSpace[np.random.choice(range(numActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0
        rolloutHeuristic1 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOnePos, getSheepPos)
        rolloutHeuristic2 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoPos, getSheepPos)
        rolloutHeuristic3 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfThreePos, getSheepPos)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state) + rolloutHeuristic3(state)) / 3

        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                           rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfOnePolicy(state), wolfTwoPolicy(state), wolfThreePolicy(state)]

        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        # prepare render
        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red'], THECOLORS['red']]
            circleSize = 10
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            saveImage = False
            saveImageDir = None
            render = Render(numOfAgent, positionIndex,
                            screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseActionList, render, renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:', len(trajectories[0]))
        print('timeTaken:', finshedTime)


if __name__ == "__main__":
    main()
