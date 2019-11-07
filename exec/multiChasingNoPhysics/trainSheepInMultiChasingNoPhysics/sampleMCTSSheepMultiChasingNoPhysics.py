import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
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


from src.episode import chooseGreedyAction,SampleTrajectory



from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle
def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # agentId = int(parametersForTrajectoryPath['agentId'])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)


    ##test
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=1
    ##test

    killzoneRadius = 30
    numSimulations = 50
    maxRunningSteps = 100
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data', 'evaluateEscapeMultiChasingNoPhysics', 'trajectoriesTest')
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

        xBoundary = [0,600]
        yBoundary = [0,600]

        #prepare render
        from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender
        import pygame as pg
        renderOn = True
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        render = Render(numOfAgent, positionIndex,
                        screen, screenColor, circleColorList, circleSize)

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
        numActionSpace = len(actionSpace)


        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))



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
        # safeBound = 80
        # wallDisToCenter = xBoundary[-1]/2
        # wallPunishRatio = 3
        # rewardFunction = reward.RewardFunctionWithWall(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal,getPreyPos)

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


        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                    rollout, backup, establishSoftmaxActionDist)

        # All agents' policies


        policy = lambda state:[sheepPolicy(state),wolfOnePolicy(state),wolfTwoPolicy(state)]

        # sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction,render,renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('lenght:',len(trajectories[0]))

        print('timeTaken:',finshedTime)

if __name__ == "__main__":
    main()