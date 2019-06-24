import sys
sys.path.append('../src')
sys.path.append('../src/constrainedChasingEscapingEnv')
import numpy as np
import pygame as pg

# local
import envNoPhysics as env

import reward

from algorithms.mcts import MCTS, CalculateScore, selectGreedyAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren
from wrapperFunctions import GetAgentPosFromState
from policies import HeatSeekingDiscreteDeterministicPolicy
from measurementFunctions import computeDistance
from analyticGeometryFunctions import computeAngleBetweenVectors


class Render():
    def __init__(self, numOfAgent, numPosEachAgent, positionIndex, screen, screenColor, circleColorList, circleSize):
        self.numOfAgent = numOfAgent
        self.numPosEachAgent = numPosEachAgent
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
                agentPos = state[i][self.positionIndex:self.positionIndex +
                                    self.numPosEachAgent]
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

            if renderOn:
                render(state)
            action = policy(state)
            trajectory.append((state, action))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


if __name__ == '__main__':

    numOfAgent = 2
    numOfOneAgentState = 2
    maxRunningSteps = 200

    sheepId = 0
    wolfId = 1
    positionIndex = 0
    numPosEachAgent = 2
    minDistance = 25

    xBoundary = [0, 640]
    yBoundary = [0, 480]

    # initPosition = np.array([[30, 30], [200, 200]])
    initPosition = np.array([[np.random.uniform(xBoundary[0], xBoundary[1]), np.random.uniform(yBoundary[0], yBoundary[1])], [np.random.uniform(xBoundary[0], xBoundary[1]), np.random.uniform(yBoundary[0], yBoundary[1])]])
    initPositionNoise = [0, 0]

    renderOn = True
    from pygame.color import THECOLORS
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, numOfOneAgentState, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex, numPosEachAgent)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex, numPosEachAgent)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPreyPos, getPredatorPos, minDistance, computeDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getPreyPos, getPredatorPos, computeAngleBetweenVectors)

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

    sampleTraj = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, render, renderOn)

    # generate trajectories
    traj = sampleTraj(policy)
    print(traj)
