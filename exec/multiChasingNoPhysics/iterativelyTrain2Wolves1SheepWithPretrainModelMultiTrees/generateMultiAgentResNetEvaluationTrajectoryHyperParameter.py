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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateStateWithCenterControlAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
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

    numTrajectoriesPerIteration = parametersForTrajectoryPath['numTrajectoriesPerIteration']
    numTrainStepEachIteration = parametersForTrajectoryPath['numTrainStepEachIteration']
    selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    otherIteration = int(parametersForTrajectoryPath['otherIteration'])

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrain2Wolves1SheepWithPretrainModelMultiTrees', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'

    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 50
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Init NoPhysics Env
        numOfAgent = 3
        sheepId = 0
        wolvesId = 1

        agentIds = list(range(numOfAgent))
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

        centerControlIndexList = [wolvesId]
        unpackAction = UnpackCenterControlAction(centerControlIndexList)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateStateWithCenterControlAction(numFramesToInterpolate, transitionFunction, isTerminal, unpackCenterControlAction)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))
        actionSpaceList = [sheepActionSpace, wolvesActionSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolvesActionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        generateModelList = [generateSheepModel, generateWolvesModel]

        sheepDepth = 9
        wolfDepth = 9
        depthList = [sheepDepth, wolfDepth]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        trainableAgentIds = [sheepId, wolvesId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    # load Model save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrain2Wolves1SheepWithPretrainModelMultiTrees', 'NNModelRes')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        # wolves policy
        if otherIteration == -999:
            # Heat Seeking policy
            wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionOneSpace, getWolfOneXPos, getSheepXPos, computeAngleBetweenVectors)
            wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionTwoSpace, getWolfTwoXPos, getSheepXPos, computeAngleBetweenVectors)
            wolfPolicy = lambda state: {(chooseGreedyAction(wolfOnePolicy(state)), chooseGreedyAction(wolfTwoPolicy(state))): 1}
        else:
            wolfModelPath = generateNNModelSavePath({'iterationIndex': otherIteration, 'agentId': wolvesId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
            restoredNNModel = restoreVariables(multiAgentNNmodel[wolvesId], wolfModelPath)
            multiAgentNNmodel[wolvesId] = restoredNNModel
            wolfPolicy = ApproximatePolicy(multiAgentNNmodel[wolvesId], wolvesActionSpace)

        # sheep policy
        sheepModelPath = generateNNModelSavePath({'iterationIndex': selfIteration, 'agentId': sheepId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
        sheepTrainedModel = restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        # sample and save trajectories
        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]

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
