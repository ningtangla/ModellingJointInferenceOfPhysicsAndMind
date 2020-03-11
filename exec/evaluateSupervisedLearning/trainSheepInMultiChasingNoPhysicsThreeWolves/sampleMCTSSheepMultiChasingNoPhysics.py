import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


from src.episode import chooseGreedyAction, SampleTrajectory, Render, SampleTrajectoryWithRender


from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


def main():
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # test
    # parametersForTrajectoryPath = {}
    # startSampleIndex = 0
    # endSampleIndex = 1
    # test

    killzoneRadius = 30
    numSimulations = 100
    maxRunningSteps = 100
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeThreeWolves', 'trajectories')
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

        # prepare render
        import pygame as pg
        renderOn = True
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red'], THECOLORS['red']]
        circleSize = 10
        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        saveImage = False
        saveImageDir = None
        render = Render(numOfAgent, positionIndex,
                        screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorOnePos = GetAgentPosFromState(wolfOneId, positionIndex)
        getPredatorTwoPos = GetAgentPosFromState(wolfTwoId, positionIndex)
        getPredatorThreePos = GetAgentPosFromState(wolfThreeId, positionIndex)

        isTerminalOne = env.IsTerminal(getPredatorOnePos, getPreyPos, killzoneRadius)
        isTerminalTwo = env.IsTerminal(getPredatorTwoPos, getPreyPos, killzoneRadius)
        isTerminalThree = env.IsTerminal(getPredatorThreePos, getPreyPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state) or isTerminalThree(state)

        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        numActionSpace = len(actionSpace)

        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorOnePos, getPreyPos, computeAngleBetweenVectors)

        wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorTwoPos, getPreyPos, computeAngleBetweenVectors)

        wolfThreePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorThreePos, getPreyPos, computeAngleBetweenVectors)
        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy

        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolfOnePolicy(state)), chooseGreedyAction(wolfTwoPolicy(state)), chooseGreedyAction(wolfThreePolicy(state))])

        # reward function

        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = reward.RewardFunctionCompete(
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
            rolloutHeuristicWeight, getPredatorOnePos, getPreyPos)
        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                           rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfOnePolicy(state), wolfTwoPolicy(state), wolfThreePolicy(state)]

        # sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction, render, renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:', len(trajectories[0]))

        print('timeTaken:', finshedTime)


if __name__ == "__main__":
    main()
