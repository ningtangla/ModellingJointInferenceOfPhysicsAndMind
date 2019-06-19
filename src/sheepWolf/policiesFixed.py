import numpy as np

def stationaryAgentPolicy(worldState):
    return (0, 0)


class PolicyActionDirectlyTowardsOtherAgent:
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


