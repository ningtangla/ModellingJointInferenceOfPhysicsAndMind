import numpy as np 
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

class HeatSeekingPolicy:
    def __init__(self, rationalityParam, actHeatSeeking, locatePredator, locatePrey):
        self.rationalityParam = rationalityParam
        self.actHeatSeeking = actHeatSeeking
        self.locatePredator = locatePredator
        self.locatePrey = locatePrey

    def __call__(self, state):
        predatorPosition = self.locatePredator(state)
        preyPosition = self.locatePrey(state)

        heatSeekingDirection = np.array(preyPosition) - np.array(predatorPosition)
        chosenActions, unchosenActions = self.actHeatSeeking(heatSeekingDirection)

        chosenActionsLikelihood = {action: self.rationalityParam / len(chosenActions) for action in chosenActions}
        unchosenActionsLikelihood = {action: (1 - self.rationalityParam) / len(unchosenActions) for action in unchosenActions}

        heatSeekingActionLikelihood = {**chosenActionsLikelihood, **unchosenActionsLikelihood}
        heatSeekingSampleLikelihood = list(heatSeekingActionLikelihood.values())
        heatSeekingActionIndex = list(np.random.multinomial(1, heatSeekingSampleLikelihood)).index(1)
        chasingAction = list(heatSeekingActionLikelihood.keys())[heatSeekingActionIndex]
        
        return chasingAction


class RandomActionPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        randomActionPolicy = {action: 1/ len(self.actionSpace) for action in self.actionSpace}

        randomActionSampleLikelihood = list(randomActionPolicy.values())
        randomActionSampledActionIndex = list(np.random.multinomial(1, randomActionSampleLikelihood)).index(1)
        randomAction = list(randomActionPolicy.keys())[randomActionSampledActionIndex]
        return randomAction
