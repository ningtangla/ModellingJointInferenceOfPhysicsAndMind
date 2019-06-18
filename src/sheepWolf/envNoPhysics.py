import numpy as np
from envSheepChaseWolf import computeDistance


class Reset():
    def __init__(self, numOfAgent, initPosition, initPositionNoise):
        self.numOfAgent = numOfAgent
        self.initPosition = initPosition
        self.initPositionNoiseLow, self.initPositionNoiseHigh = initPositionNoise

    def __call__(self):
        initPositionNoise = np.random.uniform(
            low=-self.initPositionNoiseLow, high=self.initPositionNoiseHigh, size=self.numOfAgent)
        initState = [positions +
                     initPositionNoise for positions in self.initPosition]
        return initState


class TransitionForMultiAgent():
    def __init__(self, checkBoundaryAndAdjust):
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust

    def __call__(self, state, action):
        newState = state + np.array(action)
        checkedNewStateAndVelocities = [self.checkBoundaryAndAdjust(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState


class IsTerminal():
    def __init__(self, getAgentPos, getTargetPos, minDistance):
        self.getAgentPos = getAgentPos
        self.getTargetPos = getTargetPos
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        agentPosition = self.getAgentPos(state)
        targetPosition = self.getTargetPos(state)
        if computeDistance(np.array(agentPosition), np.array(targetPosition)) <= self.minDistance:
            terminal = True
        return terminal


class CheckBoundaryAndAdjust():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        xPos, yPos = position
        if xPos >= self.xMax or xPos <= self.xMin:
            return False
        elif yPos >= self.yMax or yPos <= self.yMin:
            return False
        return True
