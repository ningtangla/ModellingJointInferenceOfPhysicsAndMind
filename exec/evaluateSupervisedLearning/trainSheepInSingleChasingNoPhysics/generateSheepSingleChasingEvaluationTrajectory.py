import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.neuralNetwork.policyValueNet import GenerateModel,ApproximatePolicy, restoreVariables

from src.episode import chooseGreedyAction,SampleTrajectory



from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle
def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # trainSteps = int(parametersForTrajectoryPath['trainSteps'])
    # parametersForTrajectoryPath['trainSteps']=trainSteps
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=100
    trainSteps=50000




    killzoneRadius = 30
    numSimulations = 100 #100
    maxRunningSteps = 150
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'evaluateSheepTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)



    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 2
        sheepId = 0
        wolfId = 1

        positionIndex = [0, 1]

        xBoundary = [0,600]
        yBoundary = [0,600]

        # prepare render
        from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender
        import pygame as pg
        renderOn = False
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        render = Render(numOfAgent, positionIndex,screen, screenColor, circleColorList, circleSize)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)

        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

        playKillzoneRadius=20
        isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, playKillzoneRadius)

        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)


        preyPowerRatio = 1.1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))



        wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)



        numStateSpace = 4#18
        regularizationFactor = 1e-4
        miniBatchSize=256
        depth=4
        learningRate=1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)

        NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'miniBatchSize':miniBatchSize,'learningRate':learningRate,'depth':depth}

        NNModelSaveDirectory = os.path.join(dirName, '..','..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trainedModels')
        NNModelSaveExtension=''
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)
        sheepTrainedModelPath = getNNModelSavePath({'trainSteps':trainSteps})

        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)

        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        # All agents' policies
        policy = lambda state:[sheepPolicy(state),wolf1Policy(state),]


        # sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        playRunninSteps=250
        sampleTrajectory = SampleTrajectoryWithRender(playRunninSteps, transitionFunction, isTerminal, reset, chooseGreedyAction,render,renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print(finshedTime)

if __name__ == "__main__":
    main()