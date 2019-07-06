import numpy as np


class UniformPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        likelihood = {action: 1/len(self.actionSpace) for action in self.actionSpace}
        return likelihood


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



class WolfPolicy:
    def __init__(self, getAgentPosition, heatSeekingActionLikelihood):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingActionLikelihood = heatSeekingActionLikelihood

    def __call__(self, mind, state, allAgentsAction):
        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')
        getWolfPos = self.getAgentPosition(wolfID)
        getSheepPos = self.getAgentPosition(sheepID)

        getWolfActionLikelihood = self.heatSeekingActionLikelihood(getWolfPos, getSheepPos)
        wolfActionLikelihood = getWolfActionLikelihood(state)

        wolfAction = allAgentsAction[wolfID]
        wolfActionProb = wolfActionLikelihood[wolfAction]

        return wolfActionProb


class SheepPolicy:
    def __init__(self, getAgentPosition, heatSeekingActionLikelihood):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingActionLikelihood = heatSeekingActionLikelihood

    def __call__(self, mind, state, allAgentsAction):
        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        getWolfPos = self.getAgentPosition(wolfID)
        getSheepPos = self.getAgentPosition(sheepID)

        getSheepActionLikelihood = self.heatSeekingActionLikelihood(getWolfPos, getSheepPos)
        sheepActionLikelihood = getSheepActionLikelihood(state)

        sheepAction = allAgentsAction[sheepID]
        sheepActionProb = sheepActionLikelihood[sheepAction]

        return sheepActionProb


class MasterPolicy:
    def __init__(self, getRandomActionLikelihood):
        self.getRandomActionLikelihood = getRandomActionLikelihood

    def __call__(self, mind, state, allAgentsAction):
        masterID = mind.index('master')
        masterActionLikelihood = self.getRandomActionLikelihood(state)
        masterAction = allAgentsAction[masterID]
        masterActionProb = masterActionLikelihood[masterAction]

        return masterActionProb
