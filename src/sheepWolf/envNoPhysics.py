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
    def __init__(self, numSimulationFrames, checkBoundaryAndAdjust, isTerminal, render, renderOn):
        self.numSimulationFrames = numSimulationFrames
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
        self.isTerminal = isTerminal
        self.render = render
        self.renderOn = renderOn

    def __call__(self, state, action):
        for i in range(self.numSimulationFrames):
            if self.isTerminal(state):
                break
            if self.renderOn:
                render(state)
            newState = np.array(state) + np.array(action)
            checkedNewStateAndVelocities = [self.checkBoundaryAndAdjust(
                position, velocity) for position, velocity in zip(newState, action)]
            newState, newAction = list(zip(*checkedNewStateAndVelocities))
            newState = [list(s) for s in newState]
        return newState


def euclideanDistance(pos1, pos2):
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


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


class GetEachState():
    def __init__(self, sheepId, wolfId):
        self.sheepId = sheepId
        self.wolfId = wolfId

    def __call__(self, state):
        sheepState = state[self.sheepId]
        wolfState = state[self.wolfId]
        return sheepState, wolfState


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
    else:
        innerProduct = vectoriseInnerProduct
    angle = np.arccos(
        innerProduct / (computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle


def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(axis=0), 0.5)


class WolfHeatSeekingPolicy:
    def __init__(self, actionSpace, getEachState):
        self.actionSpace = actionSpace
        self.getEachState = getEachState

    def __call__(self, state):
        sheepState, wolfState = self.getEachState(state)
        relativeVector = np.array(sheepState) - np.array(wolfState)
        angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
                               np.array(self.actionSpace)}
        action = angleBetweenVectors[min(angleBetweenVectors.keys())]
        return action


class SheepRandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionIndex = np.random.randint(len(actionSpace))
        action = actionSpace[actionIndex]
        return action
