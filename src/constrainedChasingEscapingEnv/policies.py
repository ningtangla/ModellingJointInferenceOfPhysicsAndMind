import numpy as np

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

class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getSelfXPos, getOtherXPos, actionMagnitude):
        self.getSelfXPos = getSelfXPos
        self.getOtherXPos = getOtherXPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        selfXPos = self.getSelfXPos(state)
        otherXPos = self.getOtherXPos(state)

        action = otherXPos - selfXPos
        actionNorm = np.sum(np.abs(action))
        if actionNorm != 0:
            action = action/actionNorm
            action *= self.actionMagnitude
        return action
