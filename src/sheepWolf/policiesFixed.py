import numpy as np

from measurementFunctions import computeAngleBetweenVectors

def stationaryAgentPolicy(worldState):
    return (0, 0)


class PolicyDirectlyTowardsOtherAgent:
    def __init__(self, getPreyXPos, getPredatorXPos, actionMagnitude):
        self.getPreyXPos = getPreyXPos
        self.getPredatorXPos = getPredatorXPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        preyXPos = self.getPreyXPos(state)
        predatorXPos = self.getPredatorXPos(state)

        action = preyXPos - predatorXPos
        actionNorm = np.linalg.norm(action, 2)
        if actionNorm != 0:
            action = action/actionNorm
            action *= self.actionMagnitude

        return action


class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace, getAgentPos, getTargetPos):
        self.actionSpace = actionSpace
        self.getAgentPos = getAgentPos
        self.getTargetPos = getTargetPos

    def __call__(self, state):
        sheepPosition = self.getAgentPos(state)
        wolfPosition = self.getTargetPos(state)
        relativeVector = np.array(sheepPosition) - np.array(wolfPosition)
        angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
                               np.array(self.actionSpace)}
        action = angleBetweenVectors[min(angleBetweenVectors.keys())]
        return action
