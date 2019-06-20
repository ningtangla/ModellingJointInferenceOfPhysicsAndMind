import numpy as np
# import env

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        return reward