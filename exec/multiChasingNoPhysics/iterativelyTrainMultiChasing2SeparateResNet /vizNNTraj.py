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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransiteForNoPhysics
TransiteForNoPhysics
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
from src.episode import SampleTrajectoryWithRender, SampleAction, chooseGreedyAction, Render, chooseGreedyAction, SelectSoftmaxAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def main():
    # input by subprocess
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    parametersForTrajectoryPath = {}
    startSampleIndex = 1
    endSampleIndex = 100
    parametersForTrajectoryPath['numTrajectoriesPerIteration'] = 16
    numTrainStepEachIteration = parametersForTrajectoryPath['numTrainStepEachIteration'] = 4
    parametersForTrajectoryPath['selfIteration'] = 9
    parametersForTrajectoryPath['otherIteration'] = 9
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    numTrajectoriesPerIteration = parametersForTrajectoryPath['numTrajectoriesPerIteration']
    numTrainStepEachIteration = parametersForTrajectoryPath['numTrainStepEachIteration']
    selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    otherIteration = int(parametersForTrajectoryPath['otherIteration'])

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 100
    numSimulations = 200
    killzoneRadius = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Init NoPhysics Env
        numOfAgent = 3
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2

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

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)

        def isTerminal(state): return isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        actionSpaceList = [sheepActionSpace, wolfActionOneSpace, wolfActionTwoSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolfActionOneSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        trainableAgentIds = [sheepId, wolfOneId, wolfTwoId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in agentIds]

        # load Model save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'NNModelRes')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        # wolves policy
        # sheepPreTrainModelPath=os.path.join(dirName,'preTrainModel','agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')

        if otherIteration == -999:
            # Heat Seeking policy
            wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(wolfActionOneSpace, getWolfOneXPos, getSheepXPos, computeAngleBetweenVectors)
            wolf2Policy = HeatSeekingDiscreteDeterministicPolicy(wolfActionTwoSpace, getWolfTwoXPos, getSheepXPos, computeAngleBetweenVectors)
        else:
            for wolfId in trainableAgentIds[1:]:
                wolfModelPath = generateNNModelSavePath({'iterationIndex': otherIteration, 'agentId': wolfId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
                # wolfModelPath=os.path.join(dirName,'preTrainModel','agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
                restoredNNModel = restoreVariables(multiAgentNNmodel[wolfId], wolfModelPath)
                multiAgentNNmodel[wolfId] = restoredNNModel
                wolf1Policy = ApproximatePolicy(multiAgentNNmodel[wolfOneId], wolfActionOneSpace)
                wolf2Policy = ApproximatePolicy(multiAgentNNmodel[wolfTwoId], wolfActionOneSpace)

        # sheep policy
        sheepModelPath = generateNNModelSavePath({'iterationIndex': selfIteration, 'agentId': sheepId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
        # sheepModelPath=os.path.join(dirName,'preTrainModel','agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        sheepTrainedModel = restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        def policy(state): return [sheepPolicy(state), wolf1Policy(state), wolf2Policy(state)]

        # sample and save trajectories
        softMaxBeta = 10
        softMaxAction = SelectSoftmaxAction(softMaxBeta)
        # chooseActionList = [softMaxAction, softMaxAction]

        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        render = None
        renderOn = True
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