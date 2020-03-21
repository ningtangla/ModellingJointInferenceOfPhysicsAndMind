import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables

from src.episode import chooseGreedyAction, SampleTrajectory, Render, SampleTrajectoryWithRender


from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # agentId = int(parametersForTrajectoryPath['agentId'])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # test
    parametersForTrajectoryPath = {}
    startSampleIndex = 0
    endSampleIndex = 10
    # test

    killzoneRadius = 80
    numSimulations = 100
    maxRunningSteps = 50
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    while True:
        # if not os.path.isfile(trajectorySavePath):
        numOfAgent = 3
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        positionIndex = [0, 1]

        xBoundary = [0, 600]
        yBoundary = [0, 600]

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorOnePos = GetAgentPosFromState(wolfOneId, positionIndex)
        getPredatorTwoPos = GetAgentPosFromState(wolfTwoId, positionIndex)

        isTerminalOne = env.IsTerminal(getPredatorOnePos, getPreyPos, killzoneRadius)
        isTerminalTwo = env.IsTerminal(getPredatorTwoPos, getPreyPos, killzoneRadius)

        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        numActionSpace = len(actionSpace)

        preyPowerRatio = 9
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 6
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorOnePos, getPreyPos, computeAngleBetweenVectors)

        wolfTwoPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorTwoPos, getPreyPos, computeAngleBetweenVectors)

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolfActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..','..', '..', '..', 'data', '2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trainedResNNModels')
        NNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'miniBatchSize': 256, 'learningRate': 0.0001, }
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

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfOnePolicy(state), wolfTwoPolicy(state)]

        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        # prepare render
        import pygame as pg
        renderOn = True
        render = True
        if renderOn:
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red'], THECOLORS['red']]
            circleSize = 10
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            saveImage = False
            saveImageDir = None
            render = Render(numOfAgent, positionIndex,
                            screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseActionList, render, renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:', len(trajectories[0]))
        print('timeTaken:', finshedTime)


if __name__ == "__main__":
    main()
