import numpy as np


class RandomActionLikelihood:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
    def __call__(self, state):
        likelihood = {action: 1/len(self.actionSpace) for action in self.actionSpace}
        return likelihood


class ActHeatSeeking:
    def __init__(self, actionSpace, calculateAngle, lowerBoundAngle, upperBoundAngle):
        self.actionSpace = actionSpace
        self.calculateAngle = calculateAngle
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle
        
    def __call__(self, heatSeekingDirection):
        heatActionAngle = {mvmtVector: self.calculateAngle(heatSeekingDirection, mvmtVector) 
                                              for mvmtVector in self.actionSpace}

        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        movementAnglePair = zip(self.actionSpace, heatActionAngle.values())

        angleFilter = {movement: angleWithinRange(angle) for movement, angle in movementAnglePair}
        chosenActions = [action for action, index in zip(angleFilter.keys(), angleFilter.values()) if index]

        unchosenFilter = [action in chosenActions for action in self.actionSpace]
        unchosenActions = [action for action, index in zip(self.actionSpace, unchosenFilter) if not index]

        return [chosenActions, unchosenActions]  


class HeatSeekingActionLikelihood:
    def __init__(self, rationalityParam, actHeatSeeking, getPredatorPos, getPreyPos):
        self.rationalityParam = rationalityParam
        self.actHeatSeeking = actHeatSeeking
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos

    def __call__(self, state):
        predatorPosition = self.getPredatorPos(state)
        preyPosition = self.getPreyPos(state)

        heatSeekingDirection = np.array(preyPosition) - np.array(predatorPosition)
        chosenActions, unchosenActions = self.actHeatSeeking(heatSeekingDirection)

        chosenActionsLikelihood = {action: self.rationalityParam / len(chosenActions) for action in chosenActions}
        unchosenActionsLikelihood = {action: (1 - self.rationalityParam) / len(unchosenActions) for action in
                                     unchosenActions}
        heatSeekingActionLikelihood = {**chosenActionsLikelihood, **unchosenActionsLikelihood}

        return heatSeekingActionLikelihood
