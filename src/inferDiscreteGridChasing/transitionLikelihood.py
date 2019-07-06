import numpy as np 

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


class PulledForceLikelihood:
    def __init__(self, forceSpace, lowerBoundAngle, upperBoundAngle, calculateAngle):
        self.forceSpace = forceSpace
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle
        self.calculateAngle = calculateAngle
        
    def __call__(self, pullersRelativeLocation):
        
        if np.all(np.array(pullersRelativeLocation) == np.array((0, 0))):
            isZeroArray = lambda force: np.all(np.array(force) == 0)
            nonZeroForceLikelihood = {force: 0 for force in self.forceSpace if not isZeroArray(force)}
            zeroForceLikelihood = {force: 1 for force in self.forceSpace if isZeroArray(force)}
            forceDirectionLikelihood = {**nonZeroForceLikelihood, **zeroForceLikelihood}
            return forceDirectionLikelihood

        forceAndRelativeLocAngle = [self.calculateAngle(pullersRelativeLocation, forceDirection) for forceDirection in self.forceSpace]
        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        angleFilter = [angleWithinRange(angle) for angle in forceAndRelativeLocAngle]
        chosenForceDirections = [force for force, index in zip(self.forceSpace, angleFilter) if index]
        unchosenFilter = [force in chosenForceDirections for force in self.forceSpace]
        unchosenForceDirections = [force for force, index in zip(self.forceSpace, unchosenFilter) if not index]

        chosenForcesLikelihood = {force: 1 / len(chosenForceDirections) for force in chosenForceDirections}
        unchosenForcesLikelihood = {force: 0 for force in unchosenForceDirections}
        forceDirectionLikelihood = {**chosenForcesLikelihood, **unchosenForcesLikelihood}

        return forceDirectionLikelihood


class PulledTransition:
    def __init__(self, getAgentPosition, getPulledAgentForceLikelihood, stayWithinBoundary):
        self.getAgentPosition = getAgentPosition
        self.getPulledAgentForceLikelihood = getPulledAgentForceLikelihood
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, physics, state, allAgentsAction, nextState):
        pulledAgentID, pullingAgentID = np.where(np.array(physics) == 'pulled')[0]

        getPulledAgentPos = self.getAgentPosition(pulledAgentID)
        getPullingAgentPos = self.getAgentPosition(pullingAgentID)

        pulledAgentState = getPulledAgentPos(state)
        pullingAgentState = getPullingAgentPos(state)

        pullersRelativeLocation = np.array(pullingAgentState) - np.array(pulledAgentState)
        getIntendedState = lambda agentState, agentAction, agentForce: self.stayWithinBoundary(sum(np.array([agentState, agentAction, agentForce])))
        pulledAgentForceLikelihood = self.getPulledAgentForceLikelihood(pullersRelativeLocation)

        pulledAgentAction = allAgentsAction[pulledAgentID]
        forceLikelihoodPair = zip(pulledAgentForceLikelihood.keys(), pulledAgentForceLikelihood.values())
        nextStateLikelihood = {tuple(getIntendedState(pulledAgentState, pulledAgentAction, force)): likelihood for force, likelihood in forceLikelihoodPair}
        pulledAgentNextState = tuple(getPulledAgentPos(nextState))
        pulledTransitionLikelihood = nextStateLikelihood.get(pulledAgentNextState, 0)

        pulledForce = np.array(pulledAgentNextState) - np.array(pulledAgentState) - np.array(pulledAgentAction)
        pullingForce = -pulledForce
        pullingAgentAction = allAgentsAction[pullingAgentID]
        nextPullingIntendedState = self.stayWithinBoundary(sum(np.array([pullingAgentState, pullingAgentAction, pullingForce])))
        nextPullingStateLikelihood = {tuple(nextPullingIntendedState): 1}
        pullingNextState = tuple(getPullingAgentPos(nextState))
        pullingTransitionLikelihood = nextPullingStateLikelihood.get(pullingNextState, 0)

        transitionLikelihood = pulledTransitionLikelihood* pullingTransitionLikelihood

        return transitionLikelihood


class NoPullTransition:
    def __init__(self, getAgentPosition, stayWithinBoundary):
        self.getAgentPosition = getAgentPosition
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, physics, state, allAgentsAction, nextState):
        noPullAgentID = physics.index('noPull')

        getNoPullAgentPos = self.getAgentPosition(noPullAgentID)
        noPullAgentState = getNoPullAgentPos(state)
        noPullAgentAction = allAgentsAction[noPullAgentID]
        nextIntendedState = self.stayWithinBoundary(np.array(noPullAgentState) + np.array(noPullAgentAction))
        nextStateLikelihood = {tuple(nextIntendedState): 1}

        noPullAgentNextState = tuple(getNoPullAgentPos(nextState))
        transitionLikelihood = nextStateLikelihood.get(noPullAgentNextState, 0)

        return transitionLikelihood
