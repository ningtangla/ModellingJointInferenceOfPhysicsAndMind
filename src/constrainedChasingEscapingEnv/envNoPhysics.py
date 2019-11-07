import numpy as np


class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
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
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState

class UnpackCenterControlAction:
    def __init__(self, centerControlIndexList):
        self.centerControlIndexList = centerControlIndexList
    def __call__(self,centerControlAction):
        upackedAction=[]
        for index,action in enumerate(centerControlAction) :
            if index in self.centerControlIndexList:
                [upackedAction.append(subaction) for subaction in action]
            else:
                upackedAction.append(action)
        return np.array(upackedAction)
class TransiteCenterControlActionForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity,unpackCenterControlAction):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.unpackCenterControlAction=unpackCenterControlAction
    def __call__(self, state, action):
        actionFortansit=self.unpackCenterControlAction(action)
        newState = state + np.array(actionFortansit)
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, actionFortansit)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState

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

class IsTerminalWithInterpolation():
    def __init__(self, getPredatorPos, getPreyPos, minDistance,divideDegree):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance
        self.divideDegree=divideDegree
    def __call__(self, lastState,currentState):
        terminal = False

        getPositionList=lambda getPos,lastState,currentState:np.linspace(getPos(lastState),getPos(currentState),self.divideDegree,endpoint=True)

        getL2Normdistance= lambda preyPosition,predatorPosition :np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)

        preyPositionList =getPositionList(self.getPreyPos,lastState,currentState)
        predatorPositionList  = getPositionList(self.getPredatorPos,lastState,currentState)

        L2NormdistanceList =[getL2Normdistance(preyPosition,predatorPosition) for (preyPosition,predatorPosition) in zip(preyPositionList,predatorPositionList) ]

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
