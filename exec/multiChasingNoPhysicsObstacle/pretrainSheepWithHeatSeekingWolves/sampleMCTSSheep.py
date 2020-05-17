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
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryAndOutObstacleByReflectVelocity, UnpackCenterControlAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, RandomPolicy
from src.episode import Render, SampleTrajectoryWithRender, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from visualize.continuousVisualization import DrawBackgroundWithObstacle


def main():
    DEBUG = 1
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 10
        agentId = 1
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        agentId = int(parametersForTrajectoryPath['agentId'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'obstacle2wolves1sheep', 'trainSheepWithHeatSeekingWolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 200
    killzoneRadius = 90
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 3
        sheepId = 0
        wolvesId = 1

        wolfOneId = 1
        wolfTwoId = 2

        xPosIndex = [0, 1]
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        xObstacles = [[120, 220], [380, 480]]
        yObstacles = [[120, 220], [380, 480]]
        isLegal = lambda state: not(np.any([(xObstacle[0] < state[0]) and (xObstacle[1] > state[0]) and (yObstacle[0] < state[1]) and (yObstacle[1] > state[1]) for xObstacle, yObstacle in zip(xObstacles, yObstacles)]))
        reset = Reset(xBoundary, yBoundary, numOfAgent, isLegal)

        getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOnePos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

        isTerminalOne = IsTerminal(getWolfOnePos, getSheepPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoPos, getSheepPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryAndOutObstacleByReflectVelocity = StayInBoundaryAndOutObstacleByReflectVelocity(xBoundary, yBoundary, xObstacles, yObstacles)
        transit = TransiteForNoPhysics(stayInBoundaryAndOutObstacleByReflectVelocity)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = actionSpace

        preyPowerRatio = 10
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        numSheepActionSpace = len(sheepActionSpace)

        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionOneSpace, getWolfOnePos, getSheepPos, computeAngleBetweenVectors)
        wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionTwoSpace, getWolfTwoPos, getSheepPos, computeAngleBetweenVectors)

    # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)

        def sheepTransit(state, action): return transit(
            state, [action, chooseActionInMCTS(wolfOnePolicy(state)), chooseActionInMCTS(wolfTwoPolicy(state))])

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
            state): return sheepActionSpace[np.random.choice(range(numSheepActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0
        rolloutHeuristic1 = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOnePos, getSheepPos)
        rolloutHeuristic2 = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoPos, getSheepPos)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2

        maxRolloutSteps = 10
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfOnePolicy(state), wolfTwoPolicy(state)]
        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            circleSize = 10
            lineWidth = 4
            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']
            obstacleColor = THECOLORS['white']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])

            drawBackground = DrawBackgroundWithObstacle(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xObstacles, yObstacles, obstacleColor)
            render = Render(numOfAgent, xPosIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir, drawBackground)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
