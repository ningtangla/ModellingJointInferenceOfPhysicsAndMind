import numpy as np 
import math 
from random import randint


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


class SamplePulledForceDirection:
    def __init__(self, calculateAngle, forceSpace, lowerBoundAngle, upperBoundAngle):
        self.calculateAngle = calculateAngle
        self.forceSpace = forceSpace
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle
        
    def __call__(self, pullingDirection):
        
        if np.all(np.array(pullingDirection) == np.array((0,0))):
            return 0,0
        
        forceActionAngle = {mvmtVector: self.calculateAngle(pullingDirection, mvmtVector) for mvmtVector in self.forceSpace}

        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle

        forceAnglePair = zip(self.forceSpace, forceActionAngle.values())
        
        
        angleFilter = {force: angleWithinRange(angle) for force, angle in forceAnglePair}

        pulledActions = [action for action, index in zip(angleFilter.keys(), angleFilter.values()) if index]

        pulledActionsLikelihood = [1 / len(pulledActions)] * len(pulledActions)

        pulledActionSampleIndex = list(np.random.multinomial(1, pulledActionsLikelihood)).index(1)
        pulledAction = pulledActions[pulledActionSampleIndex]
        return pulledAction


def roundNumber(number):
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    return math.ceil(number)

class GetPullingForce:
    def __init__(self, adjustingParam, roundNumber):
        self.adjustingParam = adjustingParam
        self.roundNumber = roundNumber
    def __call__(self, relativeLocation):
        relativeLocationArray = np.array(relativeLocation)
        distance = np.sqrt(relativeLocationArray.dot(relativeLocationArray))
        force = self.roundNumber(distance / self.adjustingParam + 1)
        return force
    

class PulledAgentTransition:
    def __init__(self, stayWithinBoundary, samplePulledForceDirection, locatePullingAgent, locatePulledAgent, getPullingForce):
        self.stayWithinBoundary = stayWithinBoundary
        self.samplePulledForceDirection = samplePulledForceDirection
        self.locatePullingAgent = locatePullingAgent
        self.locatePulledAgent = locatePulledAgent
        self.getPullingForce = getPullingForce

    def __call__(self, action, state):
        pullingAgentState = self.locatePullingAgent(state)
        pulledAgentState = self.locatePulledAgent(state)

        pullingDirection = np.array(pullingAgentState) - np.array(pulledAgentState)
        
        pulledAction = self.samplePulledForceDirection(pullingDirection)
        pullingForce = self.getPullingForce(pullingDirection)
        pullingResultAction = np.array(pulledAction) * pullingForce
        
        nextIntendedState = np.array(pulledAgentState) + pullingResultAction + np.array(action)
        pulledAgentNextState = self.stayWithinBoundary(nextIntendedState)
        return pulledAgentNextState


class PlainTransition:
    def __init__(self, stayWithinBoundary, locateCurrentAgent):
        self.stayWithinBoundary = stayWithinBoundary
        self.locateCurrentAgent = locateCurrentAgent

    def __call__(self, action, state):
        agentCurrentState = self.locateCurrentAgent(state)
        nextIntendedState = np.array(agentCurrentState) + np.array(action)
        agentNextState = self.stayWithinBoundary(nextIntendedState)
        return agentNextState


class IsTerminal:
    def __init__(self, locateChasingAgent, locateEscapingAgent):
        self.locateChasingAgent = locateChasingAgent
        self.locateEscapingAgent = locateEscapingAgent

    def __call__(self, state):

        chasingAgentPosition = self.locateChasingAgent(state)
        escapingAgentPosition = self.locateEscapingAgent(state)

        if np.all(np.array(chasingAgentPosition) == np.array(escapingAgentPosition)):
            return True
        else:
            return False




