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


class RewardFunctionWithWall():
    def __init__(self, aliveBonus, deathPenalty,safeBound, wallDisToCenter, isTerminal, getPosition):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.isTerminal = isTerminal
        self.getPosition = getPosition

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())

        wallPunish =  - np.abs(self.deathPenalty) * np.power(np.max(0, minDisToWall - self.safeBound), 2) / np.power(self.safeBound, 2)

        return reward + wallPunish

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

