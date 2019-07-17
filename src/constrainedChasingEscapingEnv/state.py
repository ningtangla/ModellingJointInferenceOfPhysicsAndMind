import numpy as np


class GetAgentPosFromState:
    def __init__(self, agentId, posIndex):
        self.agentId = agentId
        self.posIndex = posIndex

    def __call__(self, state):
        state = np.asarray(state)
        agentPos = state[self.agentId][self.posIndex]

        return agentPos
