import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME,'..', '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product

from src.constrainedChasingEscapingEnv.envNoPhysics import  UnpackCenterControlAction, TransiteForNoPhysicsWithCenterControlAction, Reset,IsTerminal,StayInBoundaryByReflectVelocity,TransitWithInterpolateStateWithCenterControlAction

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
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist,Expand,RollOut,establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction,SampleTrajectoryWithRender
from exec.parallelComputing import GenerateTrajectoriesParallel



def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory =os.path.join(dirName, '..', '..', '..', '..', 'data', '3wolves1sheep', 'trainWolvesThreeCenterControl99', 'evaTraj')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 400
    killzoneRadius = 80
    agentId = 1
    fixedParameters = {'agentId': agentId,'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

## test
    renderOn = 1
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=500
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)


    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    # while  True:
    if not os.path.isfile(trajectorySavePath):

        numOfAgent = 4
        sheepId = 0
        wolvesId = 1

        wolfOneId = 1
        wolfTwoId = 2
        wolfThreeId = 3

        xPosIndex = [0, 1]
        xBoundary = [0,600]
        yBoundary = [0,600]

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos =GetAgentPosFromState(wolfTwoId, xPosIndex)
        getWolfThreeXPos = GetAgentPosFromState(wolfThreeId, xPosIndex)

        playKillzoneRadius = 30
        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, playKillzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, playKillzoneRadius)
        isTerminalThree = IsTerminal(getWolfThreeXPos, getSheepXPos, playKillzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state) or isTerminalThree(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
        transitionFunction = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateStateWithCenterControlAction(numFramesToInterpolate, transitionFunction, isTerminal,unpackCenterControlAction)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]
        wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]

        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionThreeSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace, wolfActionThreeSpace))

        # neural network init
        numStateSpace = 8
        numSheepActionSpace=len(sheepActionSpace)
        numWolvesActionSpace=len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '3wolves1sheep', 'trainSheepInMultiChasingNoPhysicsThreeWolves', 'trainedResNNModels')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 110,'miniBatchSize':256,'learningRate':0.0001}
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
        wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '3wolves1sheep', 'trainWolvesThreeCenterControl99', 'trainedResNNModels')
        wolfId = 1
        NNModelFixedParametersWolves = {'agentId': wolfId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'miniBatchSize':256,'learningRate':0.0001,}

        getWolfNNModelSavePath = GetSavePath(wolfNNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParametersWolves)
        wolvesTrainedModelPath = getWolfNNModelSavePath({'trainSteps':50000,'depth':depth})
        wolvesTrainedModel = restoreVariables(initWolvesNNModel, wolvesTrainedModelPath)
        wolfPolicy = ApproximatePolicy(wolvesTrainedModel, wolvesActionSpace)

        from exec.evaluateNoPhysicsEnvWithRender import Render
        import pygame as pg
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['red'], THECOLORS['red']]
        circleSize = 10
        saveImage = False
        saveImageDir = os.path.join(dirName, '..','..','..', '..', 'data','img')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        render=None
        if renderOn:
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        reset = Reset(xBoundary, yBoundary, numOfAgent)
        chooseActionList = [chooseGreedyAction,chooseGreedyAction]
        playRunningSteps = 150
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
