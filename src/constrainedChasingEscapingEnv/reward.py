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
    def __init__(self, weight, getTargetPosition, getCurrentPosition):
        self.weight = weight
        self.getTargetPosition = getTargetPosition
        self.getCurrentPosition = getCurrentPosition

    def __call__(self, state):
        terminalPosition = self.getTargetPosition(state)
        currentPosition = self.getCurrentPosition(state)

        distance = np.linalg.norm(currentPosition - terminalPosition, ord = 2)
        reward = -self.weight * distance

        return reward

