import numpy as np


class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent, isLegal=lambda state: True):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent
        self.isLegal = isLegal

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),
                      np.random.uniform(yMin, yMax)]
                     for _ in range(self.numOfAgnet)]
        while np.all([self.isLegal(state) for state in initState]) is False:
            initState = [[np.random.uniform(xMin, xMax),
                          np.random.uniform(yMin, yMax)]
                         for _ in range(self.numOfAgnet)]
        return np.array(initState)


def samplePosition(xBoundary, yBoundary):
    positionX = np.random.uniform(xBoundary[0], xBoundary[1])
    positionY = np.random.uniform(yBoundary[0], yBoundary[1])
    position = [positionX, positionY]
    return position


class RandomReset():
    def __init__(self, numOfAgent, xBoundary, yBoundary, isTerminal):
        self.numOfAgent = numOfAgent
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.isTerminal = isTerminal

    def __call__(self):
        terminal = True
        while terminal:
            initState = [samplePosition(self.xBoundary, self.yBoundary) for i in range(self.numOfAgent)]
            initState = np.array(initState)
            terminal = self.isTerminal(initState)
        return initState


class FixedReset():
    def __init__(self, initPositionList):
        self.initPositionList = initPositionList

    def __call__(self, trialIndex):
        initState = self.initPositionList[trialIndex]
        initState = np.array(initState)
        return initState


class TransiteForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, state, action):
        newState = state + np.array(action)
        checkedNewPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewPositionsAndVelocities))
        return np.array(newState), np.array(newAction)


class TransitWithInterpolateState:
    def __init__(self, numFramesToInterpolate, transite, isTerminal):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.transite = transite
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        actionForInterpolation = np.array(action) / (self.numFramesToInterpolate + 1)
        for frameIndex in range(self.numFramesToInterpolate + 1):
            nextState, nextActionForInterpolation = self.transite(state, actionForInterpolation)
            if self.isTerminal(nextState):
                break
            state = nextState
            actionForInterpolation = nextActionForInterpolation
        return np.array(nextState)


class UnpackCenterControlAction:
    def __init__(self, centerControlIndexList):
        self.centerControlIndexList = centerControlIndexList

    def __call__(self, centerControlAction):
        upackedAction = []
        for index, action in enumerate(centerControlAction):
            if index in self.centerControlIndexList:
                [upackedAction.append(subAction) for subAction in action]
            else:
                upackedAction.append(action)
        return np.array(upackedAction)


class TransiteForNoPhysicsWithCenterControlAction():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, state, action):
        newState = state + np.array(action)
        checkedNewPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewPositionsAndVelocities))
        return np.array(newState), np.array(newAction)


class TransitWithInterpolateStateWithCenterControlAction:
    def __init__(self, numFramesToInterpolate, transite, isTerminal, unpackCenterControlAction):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.transite = transite
        self.isTerminal = isTerminal
        self.unpackCenterControlAction = unpackCenterControlAction

    def __call__(self, state, action):
        actionFortansit = self.unpackCenterControlAction(action)
        actionForInterpolation = np.array(actionFortansit) / (self.numFramesToInterpolate + 1)
        for frameIndex in range(self.numFramesToInterpolate + 1):
            nextState, nextActionForInterpolation = self.transite(state, actionForInterpolation)
            if self.isTerminal(nextState):
                break
            state = nextState
            actionForInterpolation = nextActionForInterpolation
        return np.array(nextState)


class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        L2Normdistance = np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)
        if L2Normdistance <= self.minDistance:
            terminal = True
        return terminal


class IsTerminalMultiAgent():
    def __init__(self, getPredatorPosList, getPreyPosList, minDistance):
        self.getPredatorPosList = getPredatorPosList
        self.getPreyPosList = getPreyPosList
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        preyPositions = [getPreyPos(state) for getPreyPos in self.getPreyPosList]
        predatorPositions = [getPredatorPos(state) for getPredatorPos in self.getPredatorPosList]

        L2Normdistances = [np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2) for preyPosition in preyPositions for predatorPosition in predatorPositions]

        if np.any(L2Normdistances) <= self.minDistance:
            terminal = True
        return terminal


class IsTerminalWithInterpolation():
    def __init__(self, getPredatorPos, getPreyPos, minDistance, divideDegree):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance
        self.divideDegree = divideDegree

    def __call__(self, lastState, currentState):
        terminal = False

        getPositionList = lambda getPos, lastState, currentState: np.linspace(getPos(lastState), getPos(currentState), self.divideDegree, endpoint=True)

        getL2Normdistance = lambda preyPosition, predatorPosition: np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)

        preyPositionList = getPositionList(self.getPreyPos, lastState, currentState)
        predatorPositionList = getPositionList(self.getPredatorPos, lastState, currentState)

        L2NormdistanceList = [getL2Normdistance(preyPosition, predatorPosition) for (preyPosition, predatorPosition) in zip(preyPositionList, predatorPositionList)]

        if np.any(np.array(L2NormdistanceList) <= self.minDistance):
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


class StayInBoundaryAndOutObstacleByReflectVelocity():
    def __init__(self, xBoundary, yBoundary, xObstacles, yObstacles):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles

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

        for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
            xObstacleMin, xObstacleMax = xObstacle
            yObstacleMin, yObstacleMax = yObstacle
            if position[0] >= xObstacleMin and position[0] <= xObstacleMax and position[1] >= yObstacleMin and position[1] <= yObstacleMax:
                if position[0] - velocity[0] <= xObstacleMin:
                    adjustedVelX = -velocity[0]
                    adjustedX = 2 * xObstacleMin - position[0]
                if position[0] - velocity[0] >= xObstacleMax:
                    adjustedVelX = -velocity[0]
                    adjustedX = 2 * xObstacleMax - position[0]
                if position[1] - velocity[1] <= yObstacleMin:
                    adjustedVelY = -velocity[1]
                    adjustedY = 2 * yObstacleMin - position[1]
                if position[1] - velocity[1] >= yObstacleMax:
                    adjustedVelY = -velocity[1]
                    adjustedY = 2 * yObstacleMax - position[1]

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
