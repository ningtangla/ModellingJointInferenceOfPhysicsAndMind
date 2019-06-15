import numpy as np
from AnalyticGeometryFunctions import computeVectorNorm, computeAngleBetweenVectors
import pygame as pg
import os

# init variables
actionSpace = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
xBoundary = [0, 180]
yBoundary = [0, 180]
vel = 1


class OptimalPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        targetState = state[2:4]
        agentState = state[0:2]
        relativeVector = np.array(targetState) - np.array(agentState)
        angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
                               np.array(self.actionSpace)}
        action = angleBetweenVectors[min(angleBetweenVectors.keys())]
        return action


def checkBound(state, xBoundary, yBoundary):
    xMin, xMax = xBoundary
    yMin, yMax = yBoundary
    xPos, yPos = state
    if xPos >= xMax or xPos <= xMin:
        return False
    elif yPos >= yMax or yPos <= yMin:
        return False
    return True


def getEachState(state):
    return state[:2], state[2:]


class TransitionFunction():
    def __init__(self, xBoundary, yBoundary, velocity):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.velocity = velocity

    def __call__(self, state, action):
        agentState, targetPosition = getEachState(state)
        actionMagnitude = computeVectorNorm(np.array(action))
        modifiedAction = np.array(action) * self.velocity / actionMagnitude
        newAgentState = np.array(agentState) + modifiedAction
        if checkBound(newAgentState, self.xBoundary, self.yBoundary):
            return np.concatenate([newAgentState, targetPosition])
        return np.concatenate([agentState, targetPosition])


class IsTerminal():
    def __init__(self, minDistance):
        self.minDistance = minDistance
        return

    def __call__(self, state):
        agentState, targetPosition = getEachState(state)
        relativeVector = np.array(agentState) - np.array(targetPosition)
        relativeDistance = computeVectorNorm(relativeVector)
        if relativeDistance <= self.minDistance:
            return True
        return False


class Reset():
    def __init__(self, xBoundary, yBoundary):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
        targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
        while not (checkBound(initialAgentState, self.xBoundary, self.yBoundary) and checkBound(targetPosition,
                                                                                                self.xBoundary,
                                                                                                self.yBoundary)):
            initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
            targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
        return np.concatenate([initialAgentState, targetPosition])


class FixedReset():
    def __init__(self, xBoundary, yBoundary):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
        targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
        initialDistance = computeVectorNorm(targetPosition - initialAgentState)
        while not (checkBound(initialAgentState, self.xBoundary, self.yBoundary) and checkBound(targetPosition,
                                                                                                self.xBoundary,
                                                                                                self.yBoundary) and initialDistance >= 20):
            initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
            targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
            initialDistance = computeVectorNorm(targetPosition - initialAgentState)
        return np.concatenate([initialAgentState, targetPosition])


class Render():
    def __init__(self, numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,
                 saveImage, saveImagePath):
        self.numAgent = numAgent
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImagePath = saveImagePath

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)
            for i in range(self.numAgent):
                oneAgentState = state[self.numOneAgentState * i: self.numOneAgentState * (i + 1)]
                oneAgentPosition = oneAgentState[min(self.positionIndex): max(self.positionIndex) + 1]
                pg.draw.circle(self.screen, self.circleColorList[i],
                               [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], self.circleSize)
            pg.display.flip()
            if self.saveImage == True:
                filenameList = os.listdir(self.saveImagePath)
                pg.image.save(self.screen, self.saveImagePath + '/' + str(len(filenameList)) + '.png')
            pg.time.wait(1)
