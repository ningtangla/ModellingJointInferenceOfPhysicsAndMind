import numpy as np


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


class HeuristicDistanceToTarget:
    def __init__(self, weight, getPredatorPosition, getPreyPosition):
        self.weight = weight
        self.getPredatorPosition = getPredatorPosition
        self.getPreyPosition = getPreyPosition

    def __call__(self, state):
        predatorPos = self.getPredatorPosition(state)
        preyPos = self.getPreyPosition(state)

        distance = np.linalg.norm(predatorPos - preyPos, ord = 2)
        reward = -self.weight * distance

        return reward
