import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import pygame as pg
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


from src.episode import chooseGreedyAction,SampleTrajectory

from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender

from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle
def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])

    # agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath={}
    startSampleIndex=0
    endSampleIndex=20
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)



    killzoneRadius = 20
    numSimulations = 100 #100
    maxRunningSteps = 60
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data',
                                             'evaluateEscapeMultiChasingNoPhysics', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)



    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 3
        sheepId = 0
        wolfId = 1
        wolf2Id = 2
        positionIndex = [0, 1]

        xBoundary = [0,600]
        yBoundary = [0,600]

        #prepare render
        # renderOn = True
        # from pygame.color import THECOLORS
        # screenColor = THECOLORS['black']
        # circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        # circleSize = 10
        # screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        # render = Render(numOfAgent, positionIndex,
        #                 screen, screenColor, circleColorList, circleSize)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)
        getPredator2Pos=GetAgentPosFromState(wolf2Id, positionIndex)
        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)


        isTerminal1 = env.IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)
        isTerminal2 =env.IsTerminal(getPredator2Pos, getPreyPos, killzoneRadius)

        isTerminal=lambda state:isTerminal1(state) or isTerminal1(state)

        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)


        preyPowerRatio = 1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))



        wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

        wolf2Policy=HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredator2Pos, getPreyPos, computeAngleBetweenVectors)
        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy

        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolf1Policy(state)), chooseGreedyAction(wolf2Policy(state))])

        # reward function
        maxRolloutSteps = 10

        aliveBonus = 1 / maxRolloutSteps
        deathPenalty = -1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

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
            rolloutHeuristicWeight, getPredatorPos, getPreyPos)

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                          rewardFunction, isTerminal, rolloutHeuristic)

        numSimulations = 200
        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                    rollout, backup, establishSoftmaxActionDist)

        # All agents' policies


        policy = lambda state:[sheepPolicy(state),wolf1Policy(state),wolf2Policy(state)]


        sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        # sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction,render,renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print(finshedTime)

if __name__ == "__main__":
    main()