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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction

import src.constrainedChasingEscapingEnv.reward as reward

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
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleAction, chooseGreedyAction, SampleTrajectoryWithRender
from exec.parallelComputing import GenerateTrajectoriesParallel


def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateSupervisedLearning', 'trainCenterControl3wolvesActionSpace55', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 100
    numSimulations = 400
    killzoneRadius = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=100
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        numOfAgent = 4
        sheepId = 0
        wolvesId = 1

        wolfOneId = 1
        wolfTwoId = 2
        wolfThreeId = 3

        xPosIndex = [0, 1]
        xBoundary = [0, 600]
        yBoundary = [0, 600]

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)
        getWolfThreeXPos = GetAgentPosFromState(wolfThreeId, xPosIndex)

        reset = Reset(xBoundary, yBoundary, numOfAgent)

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
        isTerminalThree = IsTerminal(getWolfThreeXPos, getSheepXPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state) or isTerminalThree(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
        transit = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity, unpackCenterControlAction)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]

        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 2

        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionThreeSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace, wolfActionThreeSpace))

        actionSpaceList = [sheepActionSpace, wolvesActionSpace]

        # neural network init
        numStateSpace = 8
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeThreeWolves', 'trainedResNNModels')
        NNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 100, 'numSimulations': 100, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        sheepTrainedModelPath = getNNModelSavePath({'trainSteps': 50000, 'depth': depth})
        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        from exec.evaluateNoPhysicsEnvWithRender import Render
        import pygame as pg
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red'], THECOLORS['red']]
        circleSize = 10

        saveImage = False
        saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        renderOn = False
        render = None
        if renderOn:
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)
        chooseActionList = [chooseActionInMCTS, chooseActionInMCTS]
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)

        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(wolvesActionSpace) for action in wolvesActionSpace}

    # load chase nn policy

        def wolvesTransit(state, action): return transit(
            state, [chooseGreedyAction(sheepPolicy(state)), action])

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
        rolloutHeuristicWeight = 0.1
        rolloutHeuristic1 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos)
        rolloutHeuristic2 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos)
        rolloutHeuristic3 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfThreeXPos, getSheepXPos)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state) + rolloutHeuristic3(state)) / 2

        maxRolloutSteps = 10
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
