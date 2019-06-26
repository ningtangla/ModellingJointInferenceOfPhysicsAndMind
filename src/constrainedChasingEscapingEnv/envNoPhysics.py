import numpy as np


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


class TransiteForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, state, action):
        newState = state + np.array(action)
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState


class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance, computeDistance):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance
        self.computeDistance = computeDistance

    def __call__(self, state):
        terminal = False
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        if self.computeDistance(np.array(preyPosition) - np.array(predatorPosition)) <= self.minDistance:
            terminal = True
        return terminal


class StayInBoundaryByReflectVelocity():
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
