import numpy as np


class ActHeatSeeking:
    def __init__(self, actionSpace, lowerBoundAngle, upperBoundAngle, calculateAngle):
        self.actionSpace = actionSpace
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle
        self.calculateAngle = calculateAngle

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
    def __init__(self, rationalityParam, actHeatSeeking):
        self.rationalityParam = rationalityParam
        self.actHeatSeeking = actHeatSeeking

    def __call__(self, heatSeekingDirection):
        chosenActions, unchosenActions = self.actHeatSeeking(heatSeekingDirection)
        chosenActionsLik = {action: self.rationalityParam / len(chosenActions) for action in chosenActions}
        unchosenActionsLik = {action: (1 - self.rationalityParam) / len(unchosenActions) for action in unchosenActions}
        heatSeekingActionLik = {**chosenActionsLik, **unchosenActionsLik}

        return heatSeekingActionLik


class WolfPolicy:
    def __init__(self, getAgentPosition, heatSeekingPolicy):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingPolicy = heatSeekingPolicy

    def __call__(self, mind, state, allAgentsAction):
        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        wolfPos = self.getAgentPosition(wolfID, state)
        sheepPos = self.getAgentPosition(sheepID, state)

        heatSeekingDirection = np.array(sheepPos) - np.array(wolfPos)
        wolfActionDist = self.heatSeekingPolicy(heatSeekingDirection)

        wolfAction = allAgentsAction[wolfID]
        wolfActionProb = wolfActionDist.get(wolfAction, 0)

        return wolfActionProb


class SheepPolicy:
    def __init__(self, getAgentPosition, heatSeekingPolicy):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingPolicy = heatSeekingPolicy

    def __call__(self, mind, state, allAgentsAction):
        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        wolfPos = self.getAgentPosition(wolfID, state)
        sheepPos = self.getAgentPosition(sheepID, state)

        heatSeekingDirection = np.array(sheepPos) - np.array(wolfPos)
        sheepActionDist = self.heatSeekingPolicy(heatSeekingDirection)

        sheepAction = allAgentsAction[sheepID]
        sheepActionProb = sheepActionDist.get(sheepAction, 0)

        return sheepActionProb


class MasterPolicy:
    def __init__(self, uniformPolicy):
        self.uniformPolicy = uniformPolicy

    def __call__(self, mind, state, allAgentsAction):
        masterID = mind.index('master')
        masterActionLik = self.uniformPolicy(state)
        masterAction = allAgentsAction[masterID]
        masterActionProb = masterActionLik.get(masterAction, 0)

        return masterActionProb


class UniformPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        likelihood = {action: 1/len(self.actionSpace) for action in self.actionSpace}
        return likelihood