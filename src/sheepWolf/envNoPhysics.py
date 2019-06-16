import numpy as np


class Reset():
    def __init__(self, numOfAgent, initPosition, initPositionNoise):
        self.numOfAgent = numOfAgent
        self.initPosition = initPosition  # [[],[]]
        self.initPositionNoiseLow, self.initPositionNoiseHigh = initPositionNoise

    def __call__(self):
        initPositionNoise = np.random.uniform(
            low=-self.initPositionNoiseLow, high=self.initPositionNoiseHigh, size=self.numOfAgent)
        initState = [np.asarray(
            positions) + initPositionNoise for positions in self.initPosition]
        initState = [list(s) for s in initState]
        return initState


class TransitionForMultiAgent():
    def __init__(self, checkBoundaryAndAdjust):
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust

    def __call__(self, state, action):
        newState = np.array(state) + np.array(action)
        checkedNewStateAndVelocities = [self.checkBoundaryAndAdjust(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        newState = [list(s) for s in newState]
        return newState


def euclideanDistance(pos1, pos2):
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


class IsTerminal():
    def __init__(self, sheepId, wolfId, minDistance):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        sheepPosition = state[self.sheepId]
        wolfPosition = state[self.wolfId]
        if euclideanDistance(sheepPosition, wolfPosition) <= self.minDistance:
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


if __name__ == '__main__':
    xBoundary = [0, 640]
    yBoundary = [0, 480]
    checkBoundaryAndAdjust = CheckBoundaryAndAdjust(xBoundary, yBoundary)
    state = [1, -3]
    action = [1, 1]
    print(checkBoundaryAndAdjust(state, action))
