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
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import Render, SampleTrajectoryWithRender, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from visualize.continuousVisualization import DrawBackgroundWithObstacle
from visualize.continuousVisualization import DrawBackgroundWithObstacle


def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory =os.path.join(dirName, '..', '..', '..', 'data', '2wolves1sheep', 'trainWolvesTwoCenterControl', 'evaTraj')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 300
    killzoneRadius = 90
    agentId = 1
    fixedParameters = {'agentId':agentId,'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

## test
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=499
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    renderOn = 1

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    #while  True:
    if not os.path.isfile(trajectorySavePath):

        numOfAgent = 3
        sheepId = 0
        wolvesId = 1

        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]
        xBoundary = [0,600]
        yBoundary = [0,600]
        xObstacles = [[120, 220], [380, 480]]
        yObstacles = [[120, 220], [380, 480]]
        isLegal = lambda state: not(np.any([(xObstacle[0] < state[0]) and (xObstacle[1] > state[0]) and (yObstacle[0] < state[1]) and (yObstacle[1] > state[1]) for xObstacle, yObstacle in zip(xObstacles, yObstacles)]))
        reset = Reset(xBoundary, yBoundary, numOfAgent, isLegal)

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

        playKillzoneRadius = 30 
        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, playKillzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, playKillzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        wolvesId = 1
        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
        stayInBoundaryAndOutObstacleByReflectVelocity = StayInBoundaryAndOutObstacleByReflectVelocity(xBoundary, yBoundary, xObstacles, yObstacles)
        transit = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryAndOutObstacleByReflectVelocity, unpackCenterControlAction)


        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolvesActionSpace =list(product(wolfActionOneSpace,wolfActionTwoSpace))

        # neural network init
        numStateSpace = 6
        numSheepActionSpace=len(sheepActionSpace)
        numWolvesActionSpace=len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'obstacle2wolves1sheep', 'trainSheepWithHeatSeekingWolves', 'trainedResNNModels')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 200,'miniBatchSize':256,'learningRate':0.0001}
        getSheepNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)

        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps':50000,'depth':depth})
        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)


        generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        initWolvesNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'obstacle2wolves1sheep', 'pretrainWolvesTwoCenterControlWithPretrainedSheep', 'trainedResNNModels')
        wolfId = 1
        NNModelFixedParametersWolves = {'agentId': wolfId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'miniBatchSize':256,'learningRate':0.0001,}

        getWolfNNModelSavePath = GetSavePath(wolfNNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParametersWolves)
        wolvesTrainedModelPath = getWolfNNModelSavePath({'trainSteps':50000,'depth':depth})
        wolvesTrainedModel = restoreVariables(initWolvesNNModel, wolvesTrainedModelPath)
        wolfPolicy = ApproximatePolicy(wolvesTrainedModel, wolvesActionSpace)

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

        chooseActionList = [chooseGreedyAction,chooseGreedyAction]

        playRunningSteps = 100
        sampleTrajectory = SampleTrajectoryWithRender(playRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        #saveToPickle(trajectories, trajectorySavePath)

        trajLen = [len(traj) for traj in trajectories]
        print(trajLen)
        print(np.mean(trajLen))


if __name__ == '__main__':
    main()
