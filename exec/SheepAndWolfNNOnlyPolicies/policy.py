import numpy as np

class AgentPolicy:
    def __init__(self, approximatePolicy):
        self.approximatePolicy = approximatePolicy

    def __call__(self, state, action):
        agentAction = self.approximatePolicy(state)
        actionProbability = 1 if np.all(agentAction == action) else 0
        return actionProbability

