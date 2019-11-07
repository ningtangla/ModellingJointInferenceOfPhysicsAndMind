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
            nonZeroForceLik = {force: 0 for force in self.forceSpace if not isZeroArray(force)}
            zeroForceLik = {force: 1 for force in self.forceSpace if isZeroArray(force)}
            forceDirectionLik = {**nonZeroForceLik, **zeroForceLik}
            return forceDirectionLik

        forceAndRelativeLocAngle = [self.calculateAngle(pullersRelativeLocation, forceDirection) for forceDirection in self.forceSpace]
        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        angleFilter = [angleWithinRange(angle) for angle in forceAndRelativeLocAngle]
        chosenForceDirections = [force for force, index in zip(self.forceSpace, angleFilter) if index]
        unchosenFilter = [force in chosenForceDirections for force in self.forceSpace]
        unchosenForceDirections = [force for force, index in zip(self.forceSpace, unchosenFilter) if not index]

        chosenForcesLik = {force: 1 / len(chosenForceDirections) for force in chosenForceDirections}
        unchosenForcesLik = {force: 0 for force in unchosenForceDirections}
        forceDirectionLik = {**chosenForcesLik, **unchosenForcesLik}

        return forceDirectionLik


class PulledTransition:
    def __init__(self, getAgentPosition, getPulledAgentForceLik, stayWithinBoundary):
        self.getAgentPosition = getAgentPosition
        self.getPulledAgentForceLik = getPulledAgentForceLik
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, physics, state, allAgentsAction, nextState):
        pulledAgentID, pullingAgentID = np.where(np.array(physics) == 'pulled')[0]

        pulledAgentPos = self.getAgentPosition(pulledAgentID, state)
        pullingAgentPos = self.getAgentPosition(pullingAgentID, state)

        pullersRelativeLocation = np.array(pullingAgentPos) - np.array(pulledAgentPos)
        getIntendedState = lambda agentState, agentAction, agentForce: self.stayWithinBoundary(
            sum(np.array([agentState, agentAction, agentForce])))
        pulledAgentForceLik = self.getPulledAgentForceLik(pullersRelativeLocation)

        pulledAgentAction = allAgentsAction[pulledAgentID]
        forceLikPair = zip(pulledAgentForceLik.keys(), pulledAgentForceLik.values())
        nextStateLik = {tuple(getIntendedState(pulledAgentPos, pulledAgentAction, force)): likelihood for
                        force, likelihood in forceLikPair}

        pulledAgentNextState = tuple(self.getAgentPosition(pulledAgentID, nextState))
        pulledTransitionLik = nextStateLik.get(pulledAgentNextState, 0)

        if pulledTransitionLik == 0:
            return 0
        else:
            chooseIndex = list(nextStateLik.keys()).index(pulledAgentNextState)
            pulledForce = np.array(list(pulledAgentForceLik.keys())[chooseIndex])
            pullingForce = -pulledForce
            pullingAgentAction = allAgentsAction[pullingAgentID]
            nextPullingIntendedState = self.stayWithinBoundary(
                sum(np.array([pullingAgentPos, pullingAgentAction, pullingForce])))
            nextPullingStateLik = {tuple(nextPullingIntendedState): 1}
            pullingNextState = tuple(self.getAgentPosition(pullingAgentID, nextState))
            pullingTransitionLik = nextPullingStateLik.get(pullingNextState, 0)
            transitionLik = pulledTransitionLik * pullingTransitionLik

            return transitionLik

class NoPullTransition:
    def __init__(self, getAgentPosition, stayWithinBoundary):
        self.getAgentPosition = getAgentPosition
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, physics, state, allAgentsAction, nextState):
        noPullAgentID = physics.index('noPull')

        noPullAgentPos = self.getAgentPosition(noPullAgentID, state)
        noPullAgentAction = allAgentsAction[noPullAgentID]
        nextIntendedState = self.stayWithinBoundary(np.array(noPullAgentPos) + np.array(noPullAgentAction))
        nextStateLik = {tuple(nextIntendedState): 1}

        noPullAgentNextState = tuple(self.getAgentPosition(noPullAgentID, nextState))
        transitionLik = nextStateLik.get(noPullAgentNextState, 0)

        return transitionLik
