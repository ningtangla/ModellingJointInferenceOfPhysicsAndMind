import sys
sys.path.append('../src')
sys.path.append('../src/sheepWolf')
import numpy as np
import pygame as pg

# local
import envNoPhysics as env
import envSheepChaseWolf as game
import reward

from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, HeuristicDistanceToTarget


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
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1),
                   (1, 1), (-1, -1), (1, -1), (-1, 1)]
    numActionSpace = len(actionSpace)

    numOfAgent = 2
    numOfOneAgentState = 2
    numSimulationFrames = 1000

    sheepId = 0
    wolfId = 1
    positionIndex = 0
    numPosEachAgent = 2
    minDistance = 25

    initPosition = np.array([[30, 30], [200, 200]])
    initPositionNoise = [0, 0]
    xBoundary = [0, 640]
    yBoundary = [0, 480]
    renderOn = True

    from pygame.color import THECOLORS

    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])

    render = Render(numOfAgent, numOfOneAgentState, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getAgentPos = game.GetAgentPos(sheepId, positionIndex, numPosEachAgent)
    getTargetPos = game.GetAgentPos(wolfId, positionIndex, numPosEachAgent)

    checkBoundaryAndAdjust = env.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getAgentPos, getTargetPos, minDistance)
    transitionFunction = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    wolfPolicy = game.HeatSeekingPolicy(actionSpace, getAgentPos, getTargetPos)
    # sheepPolicy = game.RandomPolicy(actionSpace)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    getActionPrior = GetActionPrior(actionSpace)

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
    numActionSpace = len(actionSpace)

    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # select next action
    selectNextAction = SelectNextAction(sheepTransit)

    sheepId = 0
    wolfId = 1
    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getTargetPos, getAgentPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)
    numSimulations = 10
    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                       rollout, backup, selectNextAction)

    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    sampleTraj = SampleTrajectory(
        numSimulationFrames, transitionFunction, isTerminal, reset, render, renderOn)

    traj = sampleTraj(policy)
    print(traj)
