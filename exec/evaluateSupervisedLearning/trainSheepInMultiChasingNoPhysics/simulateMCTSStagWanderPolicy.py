import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import math
import pandas as pd
import itertools as it
import pathos.multiprocessing as mp

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


from src.episode import chooseGreedyAction,SampleTrajectory



from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle

def main():
    manipulatedVariables = OrderedDict()
    numTrajectories = 100
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75*numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)
    startSampleIndexes = np.arange(0, numTrajectories, math.ceil(numTrajectories/numCmdList))
    endSampleIndexes = np.concatenate([startSampleIndexes[1:], [numTrajectories]])
    startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)

    startEndIndexesPair = [(startSampleIndex, endSampleIndex) for startSampleIndex, endSampleIndex in startEndIndexesPair]

    manipulatedVariables['startEndIndexesPair'] =  startEndIndexesPair

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)


    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(simulateOneCondition, parametersAllCondtion)

def simulateOneCondition(parameters):
    print(parameters)
    parametersForTrajectoryPath=parameters

    startEndIndexesPair = tuple(parameters['startEndIndexesPair'])

    ##test
    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=1
    ##test

    killzoneRadius = 15
    numSimulations = 100
    maxRunningSteps = 150
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data', 'evaluateEscapeMultiChasingNoPhysics', 'trajectoriesStagWander')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)


    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)


    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 3
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        positionIndex = [0, 1]

        xBoundary = [0,600]
        yBoundary = [0,600]

        #prepare render
        # from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender
        # import pygame as pg
        # renderOn = False
        # from pygame.color import THECOLORS
        # screenColor = THECOLORS['black']
        # circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        # circleSize = 10
        # screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        # saveImage = False
        # saveImageDir = ''
        # render = Render(numOfAgent, positionIndex,
        #                 screen, screenColor, circleColorList, circleSize,saveImage,saveImageDir)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorOnePos = GetAgentPosFromState(wolfOneId, positionIndex)
        getPredatorTwoPos=GetAgentPosFromState(wolfTwoId, positionIndex)
        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)


        isTerminalOne = env.IsTerminal(getPredatorOnePos, getPreyPos, killzoneRadius)
        isTerminalTwo =env.IsTerminal(getPredatorTwoPos, getPreyPos, killzoneRadius)

        isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)

        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]


        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        sheepActionSpace.append((0,0))

        predatorPowerRatio = 2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        numActionSpace = len(sheepActionSpace)

        wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorOnePos, getPreyPos, computeAngleBetweenVectors)

        wolfTwoPolicy=HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorTwoPos, getPreyPos, computeAngleBetweenVectors)
        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy

        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolfOnePolicy(state)), chooseGreedyAction(wolfTwoPolicy(state))])

        # reward function

        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # reward function with wall
        safeBound = 80
        wallDisToCenter = xBoundary[-1]/2
        wallPunishRatio = 3
        rewardFunction = reward.RewardFunctionWithWall(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal,getPreyPos)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            sheepActionSpace, sheepTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return sheepActionSpace[np.random.choice(range(numActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0
        rolloutHeuristic = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getPredatorOnePos, getPreyPos)
        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,rewardFunction, isTerminal, rolloutHeuristic)


        sheepPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies


        policy = lambda state:[sheepPolicy(state),wolfOnePolicy(state),wolfTwoPolicy(state)]

        sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        # sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction,render,renderOn)
        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startEndIndexesPair[0],startEndIndexesPair[1])]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:',len(trajectories[0]))
        print('timeTaken:',finshedTime)

if __name__ == "__main__":
    main()