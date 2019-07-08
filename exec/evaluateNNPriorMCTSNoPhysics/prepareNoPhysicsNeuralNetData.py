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
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.evaluationFunctions import GetSavePath

from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectory


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
    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]

    xBoundary = [0, 320]
    yBoundary = [0, 240]
    minDistance = 25

    # initPosition = np.array([[30, 30], [200, 200]])
    # initPositionNoise = [0, 0]
    # reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    renderOn = True
    from pygame.color import THECOLORS
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.RandomReset(numOfAgent, xBoundary, yBoundary, isTerminal)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    actionMagnitude = 6
    wolfPolicy = HeatSeekingContinuesDeterministicPolicy(getPredatorPos, getPreyPos, actionMagnitude)
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
    maxRolloutSteps = 10

    aliveBonus = 1 / maxRolloutSteps
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
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    numSimulations = 200
    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                       rollout, backup, selectGreedyAction)

    # All agents' policies
    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    # generate trajectories
    maxRunningSteps = 30
    numTrials = 5000
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, render, renderOn)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    sheepPolicyName = 'mcts'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'numTrials': numTrials, 'sheepPolicyName': sheepPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()


if __name__ == "__main__":
    main()
