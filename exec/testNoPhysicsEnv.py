import sys
sys.path.append('../src')
sys.path.append('../src/sheepWolf')
import numpy as np
from play import SampleTrajectory
import pygame as pg

# local
import envNoPhysics as env


class Render():
    def __init__(self, numOfAgent, numOneAgentState, screen, screenColor, circleColorList, circleSize):
        self.numOfAgent = numOfAgent
        self.numOneAgentState = numOneAgentState
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
            self.screen.fill(self.screenColor)
            for i in range(self.numOfAgent):
                oneAgentPosition = state[self.numOneAgentState *
                                         i: self.numOneAgentState * (i + 1)]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    oneAgentPosition[0]), np.int(oneAgentPosition[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(1)


if __name__ == '__main__':
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1),
                   (1, 1), (-1, -1), (1, -1), (-1, 1)]
    numActionSpace = len(actionSpace)

    numOfAgent = 2
    numOneAgentState = 2
    numSimulationFrames = 1000

    sheepId = 0
    wolfId = 1
    minDistance = 25

    initPosition = [30, 30, 200, 200]
    initPositionNoise = [0, 0]
    xBoundary = [0, 640]
    yBoundary = [0, 480]
    renderOn = True

    from pygame.color import THECOLORS

    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    screenColor = [THECOLORS['black']]
    circleSize = 8
    titleSize = 10
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])

    render = Render(numOfAgent, numOneAgentState, screen,
                    screenColor, circleColorList, circleSize)

    checkBoundaryAndAdjust = env.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(sheepId, wolfId, minDistance)
    transitionFunction = env.TransitionForMultiAgent(
        numSimulationFrames, checkBoundaryAndAdjust, isTerminal, render, renderOn)
    reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    getEachState = env.GetEachState(sheepId, wolfId)
    wolfPolicy = env.WolfHeatSeekingPolicy(actionSpace, getEachState)
    sheepPolicy = env.SheepRandomPolicy(actionSpace)

    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    sampleTraj = SampleTrajectory(
        numSimulationFrames, transitionFunction, isTerminal, reset)

    sampleTraj(policy)
