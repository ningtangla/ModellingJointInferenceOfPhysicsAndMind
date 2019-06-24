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

def roundNumber(number):
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    return math.ceil(number)

class GetPullingForceValue:
    def __init__(self, adjustingParam, roundNumber):
        self.adjustingParam = adjustingParam
        self.roundNumber = roundNumber
    def __call__(self, relativeLocation):
        relativeLocationArray = np.array(relativeLocation)
        distance = np.sqrt(relativeLocationArray.dot(relativeLocationArray))
        force = self.roundNumber(distance / self.adjustingParam + 1)
        return force

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

    
class GetAgentsForceAction:
    def __init__(self, locatePullingAgent, locatePulledAgent, samplePulledForceDirection, getPullingForceValue):

        self.locatePullingAgent = locatePullingAgent
        self.locatePulledAgent = locatePulledAgent
        self.samplePulledForceDirection = samplePulledForceDirection
        self.getPullingForceValue = getPullingForceValue

    def __call__(self, state):
        pullingAgentState = self.locatePullingAgent(state)
        pulledAgentState = self.locatePulledAgent(state)
        pullingDirection = np.array(pullingAgentState) - np.array(pulledAgentState)       
        pulledDirection = self.samplePulledForceDirection(pullingDirection) 
        pullingForceValue = self.getPullingForceValue(pullingDirection)

        pullingResultAction = np.array(pulledDirection) * pullingForceValue

        pullingResultDependentAction = -pullingResultAction
        noPullingAction = (0,0)   

        agentsForceAction = [pullingResultAction, noPullingAction, pullingResultDependentAction]

        return agentsForceAction


class Transition:
    def __init__(self, stayWithinBoundary, getAgentsForceAction):
        self.stayWithinBoundary = stayWithinBoundary
        self.getAgentsForceAction = getAgentsForceAction

    def __call__(self, allAgentActions, state):
        agentsForceAction = self.getAgentsForceAction(state)

        nextIntendedState = [np.array(currentState) + np.array(agentForce) + np.array(action) for currentState, agentForce, action in zip(state, agentsForceAction, allAgentActions)]

        agentsNextState = [self.stayWithinBoundary(agentIntendedState) for agentIntendedState in nextIntendedState]

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

