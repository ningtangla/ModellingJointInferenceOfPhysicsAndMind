import numpy as np
from analyticGeometryFunctions import computeAngleBetweenVectors

def stationaryAgentPolicy(state):
    return (0, 0)


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionIndex = np.random.randint(len(self.actionSpace))
        action = self.actionSpace[actionIndex]
        return action


class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace, getChaserPos, getEscaperPos):
        self.actionSpace = actionSpace
        self.getChaserPos = getChaserPos
        self.getEscaperPos = getEscaperPos

    def __call__(self, state):
        relativeVector = np.array(self.getEscaperPos(state)) - np.array(self.getChaserPos(state))
        angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
                               np.array(self.actionSpace)}
        action = angleBetweenVectors[min(angleBetweenVectors.keys())]
        return action

class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self,  getChaserPos, getEscaperPos, actionMagnitude):
        self.getChaserPos = getChaserPos
        self.getEscaperPos = getEscaperPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):

        action = np.array(self.getEscaperPos(state)) - np.array(self.getChaserPos(state))
        actionNorm = np.sum(np.abs(action))
        if actionNorm != 0:
            action = action/actionNorm
            action *= self.actionMagnitude

        return action
