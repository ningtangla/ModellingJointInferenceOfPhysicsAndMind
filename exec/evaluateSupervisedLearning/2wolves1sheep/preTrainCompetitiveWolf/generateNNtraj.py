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

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, StochasticMCTS, backup, Expand, RollOut, establishPlainActionDistFromMultipleTrees
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateState, TransitWithInterpolateStateWithCenterControlAction
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy, RandomPolicy
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.episode import SampleAction, chooseGreedyAction, Render, SampleTrajectoryWithRender
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle


def main():
    DEBUG = 1
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 500
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', '2wolves1sheep', 'preTrainCompetitiveWolf', 'evaTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 50
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        numOfAgent = 3

        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]

        getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOnePos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

        isTerminalOne = IsTerminal(getWolfOnePos, getSheepPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoPos, getSheepPos, killzoneRadius)

        def isTerminal(state): return isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateState(numFramesToInterpolate, transitionFunction, isTerminal)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = actionSpace

        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        actionSpaceList = [sheepActionSpace, wolfActionOneSpace, wolfActionTwoSpace]

        wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionTwoSpace, getWolfTwoPos, getSheepPos, computeAngleBetweenVectors)

        # neural network init
        numStateSpace = 2 * numOfAgent
        numWolfActionSpace = len(wolfActionOneSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numWolfActionSpace, regularizationFactor)
        generateWolfModel = GenerateModel(numStateSpace, numWolfActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trainedResNNModels')
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

        wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', '2wolves1sheep', 'preTrainCompetitiveWolf', 'trainedResNNModels')
        NNModelSaveExtension = ''
        wolfNNModelFixedParameters = {'agentId': 1, 'maxRunningSteps': 50, 'numSimulations': 250, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getWolfNNModelSavePath = GetSavePath(wolfNNModelSaveDirectory, NNModelSaveExtension, wolfNNModelFixedParameters)

        getWolfNNModelSavePath = GetSavePath(wolfNNModelSaveDirectory, NNModelSaveExtension, wolfNNModelFixedParameters)
        initWolfNNModel = generateWolfModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        wolfTrainedModelPath = getWolfNNModelSavePath({'trainSteps': 50000, 'depth': depth})
        wolfTrainedModel = restoreVariables(initWolfNNModel, wolfTrainedModelPath)
        wolfPolicy = ApproximatePolicy(wolfTrainedModel, wolfActionOneSpace)

        # All agents' policies
        wolfTwoPolicy = RandomPolicy(sheepActionSpace)

        def policy(state): return [sheepPolicy(state), wolfPolicy(state), wolfTwoPolicy(state)]
        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['orange'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
