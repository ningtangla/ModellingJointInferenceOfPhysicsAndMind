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


class PulledForceDirectionLikelihood:
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
