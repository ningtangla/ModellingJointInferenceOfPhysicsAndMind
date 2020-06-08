import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateStateWithCenterControlAction
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.episode import SampleAction, chooseGreedyAction, Render, ForwardOneStep, SampleTrajectory, SampleTrajectoryWithRender
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle


class ForwardOneStep:
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction

    def __call__(self, state, sampleAction):
        action = sampleAction(state)
        nextState = self.transitionFunction(state, action)
        reward = self.rewardFunction(state, action, nextState)
        return (state, action, nextState, reward)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep

    def __call__(self, sampleAction):
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 0))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState

        return trajectory


class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep
        self.render = render
        self.renderOn = renderOn

    def __call__(self, sampleAction):
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 0))
                break
            if self.renderOn:
                self.render(state)
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState

        return trajectory


def main():
    DEBUG = 1
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 100
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainWolvesTwoCenterControlAction88', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 2
    killzoneRadius = 50
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

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

        resetState = Reset(xBoundary, yBoundary, numOfAgent)

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
        transitionFunction = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateStateWithCenterControlAction(numFramesToInterpolate, transitionFunction, isTerminal, unpackCenterControlAction)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

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
        NNModelSaveExtension = ''
        sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trainedResNNModels')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 110, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getSheepNNModelSavePath = GetSavePath(sheepNNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps': 50000, 'depth': depth})
        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

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
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolvesActionSpace, wolvesTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return wolvesActionSpace[np.random.choice(range(numWolvesActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0
        minDistance = 400
        rolloutHeuristic1 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos, minDistance)
        rolloutHeuristic2 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos, minDistance)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2

        maxRolloutSteps = 15
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]

        def sampleAction(state):
            actionDists = [sheepPolicy(state), wolfPolicy(state)]
            action = [chooseAction(actionDist) for actionDist, chooseAction in zip(actionDists, chooseActionList)]
            return action

        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        forwardOneStep = ForwardOneStep(transit, rewardFunction)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn)

        trajectories = [sampleTrajectory(sampleAction) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
