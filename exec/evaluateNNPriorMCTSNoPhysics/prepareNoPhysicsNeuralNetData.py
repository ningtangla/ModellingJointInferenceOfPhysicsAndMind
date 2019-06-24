import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
import numpy as np
import pickle
import random
import pygame as pg

from src.algorithms.mcts import MCTS, CalculateScore, selectGreedyAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren

import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.evaluationFunctions import GetSavePath


class Render():
    def __init__(self, numOfAgent, numPosEachAgent, positionIndex, screen, screenColor, circleColorList, circleSize):
        self.numOfAgent = numOfAgent
        self.positionIndex = positionIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)

            for i in range(self.numOfAgent):
                agentPos = state[i][self.positionIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(1)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None))
                break

            if self.renderOn:
                self.render(state)
            action = policy(state)
            trajectory.append((state, action))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class SampleTrajectorySingleAgentActions:
    def __init__(self, sampleTrajectory, agentId):
        self.sampleTrajectory = sampleTrajectory
        self.agentId = agentId

    def __call__(self, policy):
        trajectory = self.sampleTrajectory(policy)
        trajectorySingleAgentActions = [(state, action[self.agentId]) if action is not None else (state, None)
                                        for state, action in trajectory]

        return trajectorySingleAgentActions


def generateData(sampleTrajectory, policy, actionSpace, trajNumber, path):
    totalStateBatch = []
    totalActionBatch = []
    for index in range(trajNumber):
        if index % 100 == 0:
            print(index)
        trajectory = sampleTrajectory(policy)
        states, actions = zip(*trajectory)
        totalStateBatch = totalStateBatch + list(states)
        oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in
                          range(len(actionSpace))] for action in actions]
        totalActionBatch = totalActionBatch + oneHotActions

    dataSet = list(zip(totalStateBatch, totalActionBatch))
    saveFile = open(path, "wb")
    pickle.dump(dataSet, saveFile)


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()
    return dataSet


def sampleData(data, batchSize):
    batch = random.sample(data, batchSize)
    batchInput = [x for x, _ in batch]
    batchOutput = [y for _, y in batch]
    return batchInput, batchOutput


def main():

    numOfAgent = 2
    numOfOneAgentState = 2

    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]
    minDistance = 25

    xBoundary = [0, 640]
    yBoundary = [0, 480]

    initPosition = np.array([[30, 30], [200, 200]])
    # initPosition = np.array([[np.random.uniform(xBoundary[0], xBoundary[1]),np.random.uniform(yBoundary[0], yBoundary[1])],[np.random.uniform(xBoundary[0], xBoundary[1]),np.random.uniform(yBoundary[0], yBoundary[1])]])
    initPositionNoise = [0, 0]

    renderOn = True
    from pygame.color import THECOLORS
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, numOfOneAgentState, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPreyPos, getPredatorPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    def sheepTransit(state, action): return transitionFunction(
        state, [action, wolfPolicy(state)])

    # reward function
    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # initialize children; expand
    initializeChildren = InitializeChildren(
        actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    numSimulations = 10
    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                       rollout, backup, selectGreedyAction)

    # All agents' policies
    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    # generate trajectories
    maxRunningSteps = 15
    numTrials = 10
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, render, renderOn)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    sheepPolicyName = 'mcts'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'posInit': initPosition, 'numSimulations': numSimulations,
                          'numTrials': numTrials, 'sheepPolicyName': sheepPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()


if __name__ == "__main__":
    main()
