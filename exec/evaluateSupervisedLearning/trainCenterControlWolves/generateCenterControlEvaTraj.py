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

from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysics, Reset,IsTerminal,StayInBoundaryByReflectVelocity

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
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel

class SampleTrajectoryWithRender:
    def __init__(self,maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn
        self.time = 0

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state,self.time)
            self.time += 1
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            actionFortransit=[action[0],action[1][0],action[1][1]]
            nextState = self.transit(state, actionFortransit)

            state = nextState
        return trajectory


def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentResNetNoPhysicsCenterControl', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 100
    numSimulations = 200
    killzoneRadius = 25
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

## test
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=11
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)


    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    # while  True:
    if not os.path.isfile(trajectorySavePath):

        numOfAgent=3
        sheepId = 0
        wolvesId = 1

        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]
        xBoundary = [0,600]
        yBoundary = [0,600]

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos =GetAgentPosFromState(wolfTwoId, xPosIndex)

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
        isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

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
        NNModelSaveDirectory = os.path.join(dirName, '..','..', '..', 'data', 'evaluateEscapeMultiChasingNoPhysics', 'trainedResNNModelsMultiStillAction')
        NNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 150, 'numSimulations': 200,'miniBatchSize':256,'learningRate':0.0001}
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        depth = 5
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        sheepTrainedModelPath = getNNModelSavePath({'trainSteps':50000,'depth':depth})
        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)


        generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        initWolvesNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateSupervisedLearning', 'multiMCTSAgentResNetNoPhysicsCenterControl', 'trainedResNNModels')
        wolfId = 1
        NNModelFixedParametersWolves = {'agentId': wolfId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'miniBatchSize':256,'learningRate':0.0001,}

        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParametersWolves)
        wolvesTrainedModelPath = getNNModelSavePath({'trainSteps':50000,'depth':5})
        wolvesTrainedModel = restoreVariables(initWolvesNNModel, wolvesTrainedModelPath)
        wolfPolicy = ApproximatePolicy(wolvesTrainedModel, wolvesActionSpace)

        from exec.evaluateNoPhysicsEnvWithRender import Render
        import pygame as pg
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        saveImage = True
        saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        render=None
        renderOn = True
        if renderOn:
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        reset = Reset(xBoundary, yBoundary, numOfAgent)
        chooseActionList = [chooseGreedyAction,chooseGreedyAction]
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)

        print([len(traj) for traj in trajectories])


if __name__ == '__main__':
    main()
