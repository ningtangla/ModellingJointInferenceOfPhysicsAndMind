import numpy as np
import random


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
    def __init__(self, actionSpace, getPreyPos, getPredatorPos, computeAngleBetweenVectors):
        self.actionSpace = actionSpace
        self.getPreyPos = getPreyPos
        self.getPredatorPos = getPredatorPos
        self.computeAngleBetweenVectors = computeAngleBetweenVectors

    def __call__(self, state):
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        heatSeekingVector = np.array(preyPosition) - np.array(predatorPosition)
        angleBetweenVectors = {action: self.computeAngleBetweenVectors(heatSeekingVector, np.array(action)) for action in self.actionSpace}
        optimalActionList = [action for action in angleBetweenVectors.keys() if angleBetweenVectors[action] == min(angleBetweenVectors.values())]
        action = random.choice(optimalActionList)
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
            action = action / actionNorm
            action *= self.actionMagnitude
        return action
