import numpy as np
# import env

class RewardFunctionTerminalPenalty():
    def __init__(self, sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal):
        self.sheepId = sheepId
        self.wolfId = wolfId 
        self.numOneAgentState = numOneAgentState 
        self.positionIndex = positionIndex
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        sheepState = state[self.numOneAgentState * self.sheepId : self.numOneAgentState * (self.sheepId + 1)]
        sheepPosition = sheepState[min(self.positionIndex) : max(self.positionIndex) + 1]
        wolfState = state[self.numOneAgentState * self.wolfId : self.numOneAgentState * (self.wolfId + 1)]
        wolfPosition = wolfState[min(self.positionIndex) : max(self.positionIndex) + 1]
        distanceToWolfReward = 0 * (0.01 * np.power(np.sum(np.power(sheepPosition - wolfPosition, 2)), 0.5))
        reward = distanceToWolfReward +  self.aliveBouns
        if self.isTerminal(state):
            reward = reward + self.deathPenalty
        return reward


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