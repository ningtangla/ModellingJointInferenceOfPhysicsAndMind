import os
import sys
import numpy as np
import pygame as pg
import random
from pygame.color import THECOLORS

# local
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.constrainedChasingEscapingEnv.envNoPhysics as env
from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import chooseGreedyAction


class Render():
    def __init__(self, numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize,saveImage, saveImageDir):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage  = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, state, timeStep):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)
            for i in range(self.numOfAgent):
                agentPos = state[i][self.posIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")



class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn

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
                print(state)
                self.render(state, runningStep)
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


if __name__ == '__main__':
    numOfAgent = 2
    sheepId = 0
    wolfId = 1
    posIndex = [0, 1]
    minDistance = 25

    xBoundary = [0, 640]
    yBoundary = [0, 480]

    renderOn = True
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, posIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, posIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, posIndex)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(
        xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPreyPos, getPredatorPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(
        stayInBoundaryByReflectVelocity)
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
                       (-8, 0), (-6, -6), (0, -8), (6, -6)]
    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    # actionMagnitude = 8
    # wolfPolicy = HeatSeekingContinuesDeterministicPolicy(getPredatorPos, getPreyPos, actionMagnitude)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    def sheepTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(wolfPolicy(state))])

    # reward function
    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # prior
    getActionPrior = lambda state: {
        action: 1 / len(actionSpace) for action in actionSpace}

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


    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                       rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    maxRunningSteps = 100
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction, render, renderOn)

    # generate trajectories
    numTrials = 100
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]
