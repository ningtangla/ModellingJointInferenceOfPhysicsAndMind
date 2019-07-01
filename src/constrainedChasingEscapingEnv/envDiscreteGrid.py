import numpy as np 
import math 
from random import randint
from wrapperFunctions import rearrangeList

class Reset:
    def __init__(self, gridSize, lowerGridBound, agentCount):
        self.gridX, self.gridY = gridSize
        self.lowerGridBound = lowerGridBound
        self.agentCount = agentCount

    def __call__(self):
        startState = [(randint(self.lowerGridBound, self.gridX), randint(self.lowerGridBound, self.gridY)) for _ in range(self.agentCount)]
        return startState

class StayWithinBoundary:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, nextIntendedState):
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY
        return nextX, nextY

def roundNumber(number):
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    return math.ceil(number)

class GetPullingForceValue:
    def __init__(self, adjustingParam, roundNumber):
        #name 
        self.adjustingParam = adjustingParam
        self.roundNumber = roundNumber
    def __call__(self, relativeLocation):
        relativeLocationArray = np.array(relativeLocation)
        distance = np.sqrt(relativeLocationArray.dot(relativeLocationArray))
        force = self.roundNumber(distance / self.adjustingParam + 1)
        return force

class SamplePulledForceDirection:
    def __init__(self, calculateAngle, forceSpace, lowerBoundAngle, upperBoundAngle):
        # order
        self.calculateAngle = calculateAngle
        self.forceSpace = forceSpace
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle
        
    def __call__(self, pullingDirection):
        
        if np.all(np.array(pullingDirection) == np.array((0, 0))):
            return 0, 0
        
        forceActionAngle = {mvmtVector: self.calculateAngle(pullingDirection, mvmtVector) for mvmtVector in self.forceSpace}

        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        forceAnglePair = zip(self.forceSpace, forceActionAngle.values())
       
       #simplify
        angleFilter = {force: angleWithinRange(angle) for force, angle in forceAnglePair}

        pulledActions = [action for action, index in zip(angleFilter.keys(), angleFilter.values()) if index]
        pulledActionsLikelihood = [1 / len(pulledActions)] * len(pulledActions)
        pulledActionSampleIndex = list(np.random.multinomial(1, pulledActionsLikelihood)).index(1)
        pulledAction = pulledActions[pulledActionSampleIndex]
        return pulledAction

class GetPulledAgentForce:
    def __init__(self, getPullingAgentPosition, getPulledAgentPosition, samplePulledForceDirection, getPullingForceValue):
        self.getPullingAgentPosition = getPullingAgentPosition
        self.getPulledAgentPosition = getPulledAgentPosition
        self.samplePulledForceDirection = samplePulledForceDirection
        self.getPullingForceValue = getPullingForceValue

    def __call__(self, state):
        pullingAgentState = self.getPullingAgentPosition(state)
        pulledAgentState = self.getPulledAgentPosition(state)
        pullingDirection = np.array(pullingAgentState) - np.array(pulledAgentState)
        pulledDirection = self.samplePulledForceDirection(pullingDirection)
        pullingForceValue = self.getPullingForceValue(pullingDirection)
        pullingResultAction = np.array(pulledDirection) * pullingForceValue

        return pullingResultAction

class GetAgentsForce: # ordered by index
    def __init__(self, getPulledAgentForce, pulledAgentIndex, noPullingAgentIndex, pullingAgentIndex):
        self.getPulledAgentForce = getPulledAgentForce
        self.pulledAgentIndex = pulledAgentIndex
        self.noPullingAgentIndex = noPullingAgentIndex
        self.pullingAgentIndex = pullingAgentIndex
    def __call__(self, state):
        pulledAgentForce = np.array(self.getPulledAgentForce(state))
        pullingAgentForce = -pulledAgentForce
        noPullAgentForce = (0,0)
        unorderedAgentsForce = [pulledAgentForce, noPullAgentForce, pullingAgentForce]
        agentsIDOrder = [self.pulledAgentIndex, self.noPullingAgentIndex, self.pullingAgentIndex]
        agentsForce = rearrangeList(unorderedAgentsForce, agentsIDOrder)
        return agentsForce

class Transition:
    def __init__(self, stayWithinBoundary, getAgentsForce):
        self.stayWithinBoundary = stayWithinBoundary
        self.getAgentsForce = getAgentsForce
    def __call__(self, actionList, state):
        agentsForce = self.getAgentsForce(state)
        agentsIntendedState = np.array(state) + np.array(agentsForce) + np.array(actionList)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState


class IsTerminal:
    def __init__(self, locatePredator, locatePrey):
        self.locatePredator = locatePredator
        self.locatePrey = locatePrey

    def __call__(self, state):
        predatorPosition = self.locatePredator(state)
        preyPosition = self.locatePrey(state)

        if np.all(np.array(predatorPosition) == np.array(preyPosition)):
            return True
        else:
            return False

