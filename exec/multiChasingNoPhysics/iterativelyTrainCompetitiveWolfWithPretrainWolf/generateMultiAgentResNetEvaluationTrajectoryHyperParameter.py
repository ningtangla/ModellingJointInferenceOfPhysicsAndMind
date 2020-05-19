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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateState
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, RewardFunctionForCompetition
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from src.episode import SampleTrajectoryWithRender, SampleAction, chooseGreedyAction, Render, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def main():
    # input by subprocess
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    otherIteration = int(parametersForTrajectoryPath['otherIteration'])
    sheepIteration = int(parametersForTrajectoryPath['sheepIteration'])

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrainCompetitiveWolfWithPretrainWolf', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'

    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 50
    numTrajectoriesPerIteration = 1
    numTrainStepEachIteration = 1

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        # Init NoPhysics Env
        numOfAgent = 3
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2

        posIndex = [0, 1]
        wolfOnePosIndex = 1
        wolfTwoIndex = 2

        getSheepXPos = GetAgentPosFromState(sheepId, posIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOnePosIndex, posIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoIndex, posIndex)

        xBoundary = [0, 600]
        yBoundary = [0, 600]
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        playKillzoneRadius = killzoneRadius
        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, playKillzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, playKillzoneRadius)
        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        transitionFunction = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateState(numFramesToInterpolate, transitionFunction, isTerminal)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        actionSpaceList = [sheepActionSpace, wolfActionOneSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolfActionSpace = len(wolfActionOneSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolfModel = GenerateModel(numStateSpace, numWolfActionSpace, regularizationFactor)
        generateWolfTwoModel = GenerateModel(numStateSpace, numWolfActionSpace, regularizationFactor)
        generateModelList = [generateSheepModel, generateWolfModel, generateWolfTwoModel]

        sheepDepth = 9
        wolfDepth = 9
        depthList = [sheepDepth, wolfDepth, wolfDepth]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        trainableAgentIds = [sheepId, wolfOneId, wolfTwoId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    # load Model save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrainCompetitiveWolfWithPretrainWolf', 'NNModelRes')
        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        # sheep policy
        sheepModelPath = generateNNModelSavePath({'iterationIndex': sheepIteration, 'agentId': sheepId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
        sheepTrainedModel = restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        wolfModelPath = generateNNModelSavePath({'iterationIndex': selfIteration, 'agentId': wolfOneId})
        restoredNNModel = restoreVariables(multiAgentNNmodel[wolfOneId], wolfModelPath)
        multiAgentNNmodel[wolfOneId] = restoredNNModel
        wolfPolicy = ApproximatePolicy(multiAgentNNmodel[wolfOneId], wolvesActionSpace)

        # wolves policy
        if otherIteration == -999:
            wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionTwoSpace, getWolfTwoXPos, getSheepXPos, computeAngleBetweenVectors)
        else:
            wolfTwoModelPath = generateNNModelSavePath({'iterationIndex': otherIteration, 'agentId': wolfOneId})
            restoredNNModel = restoreVariables(multiAgentNNmodel[wolfOneId], wolfTwoModelPath)
            multiAgentNNmodel[wolfTwoId] = restoredNNModel
            wolfTwoPolicy = ApproximatePolicy(multiAgentNNmodel[wolfTwoId], wolfActionTwoSpace)

        def competitorPolicy(state): return wolfTwoPolicy([state[0], state[2], state[1]])
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state), competitorPolicy(state)]
        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        render = None
        renderOn = False
        if renderOn:
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['orange']]
            circleSize = 10
            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
