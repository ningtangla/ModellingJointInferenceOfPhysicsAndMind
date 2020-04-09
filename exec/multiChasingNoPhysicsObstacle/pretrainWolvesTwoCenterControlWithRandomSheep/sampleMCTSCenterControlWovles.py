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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryAndOutObstacleByReflectVelocity, UnpackCenterControlAction
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
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy, RandomPolicy
from src.episode import Render, SampleTrajectoryWithRender, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from visualize.continuousVisualization import DrawBackgroundWithObstacle


def main():
    DEBUG = 1
    renderOn = 1
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'obstacle2wolvesRandomsheep', 'trainWolvesTwoCenterControl', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 300
    killzoneRadius = 80
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

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        wolvesId = 1
        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
        stayInBoundaryAndOutObstacleByReflectVelocity = StayInBoundaryAndOutObstacleByReflectVelocity(xBoundary, yBoundary, xObstacles, yObstacles)
        transit = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryAndOutObstacleByReflectVelocity, unpackCenterControlAction)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = actionSpace
        # wolfActionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]

        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 8

        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))

        actionSpaceList = [sheepActionSpace, wolvesActionSpace]

        # neural network init
        numStateSpace = 2 * numOfAgent
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir

        sheepPolicy = RandomPolicy(sheepActionSpace)

    # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(wolvesActionSpace) for action in wolvesActionSpace}

    # load chase nn policy
        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)

        def wolvesTransit(state, action): return transit(
            state, [chooseActionInMCTS(sheepPolicy(state)), action])

        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolvesActionSpace, wolvesTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return wolvesActionSpace[np.random.choice(range(numWolvesActionSpace))]

        # rollout
        rolloutHeuristicWeight = 1e-2
        minDistance = 400
        rolloutHeuristic1 = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos, 400)
        rolloutHeuristic2 = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos, 400)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2

        maxRolloutSteps = 5
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]

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
