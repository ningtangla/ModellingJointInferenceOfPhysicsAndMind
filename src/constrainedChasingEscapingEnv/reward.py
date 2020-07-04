import numpy as np

class RewardFunctionForCompetition():
    def __init__(self, aliveBonus, deathPenalty, isSelfTerminal, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isSelfTerminal = isSelfTerminal
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            if self.isSelfTerminal:
                reward += self.deathPenalty
            else:
                reward += -self.deathPenalty
        return reward

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
class RewardFunctionCompeteWithInterpolationTerminal():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, lastState,currentState, action):
        reward = self.aliveBonus
        if self.isTerminal(lastState,currentState):
            reward += self.deathPenalty

        return reward

class IsCollided:
    def __init__(self, minXDis, getSelfPos, getOtherPos):
        self.minXDis = minXDis
        self.getSelfPos = getSelfPos
        self.getOtherPos = getOtherPos

    def __call__(self, state):
        state = np.asarray(state)
        selfPos = self.getSelfPos(state)
        otherPositions = [getPos(state) for getPos in self.getOtherPos]

        L2Normdistance = [np.linalg.norm((selfPos - otherPosition), ord=2) for otherPosition in otherPositions]
        terminal = np.any(np.array(L2Normdistance) <= self.minXDis)

        return terminal


class RewardFunctionWithWall():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal, getPosition):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.isTerminal = isTerminal
        self.getPosition = getPosition

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish =  - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0,self.safeBound -  minDisToWall), 2) / np.power(self.safeBound, 2)

        return reward + wallPunish

class RewardFunctionAvoidCollisionAndWall():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isCollided, getPosition, getVelocity):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.velocityBound = velocityBound
        self.isCollided = isCollided
        self.getPosition = getPosition
        self.getVelocity = getVelocity

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isCollided(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish =  - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0,self.safeBound -  minDisToWall), 2) / np.power(self.safeBound, 2)

        agentVel = self.getVelocity(state)
        velPunish = -np.abs(self.aliveBonus) if np.linalg.norm(agentVel) <= self.velocityBound else 0

        return reward + wallPunish + velPunish

class HeuristicDistanceToTarget:
    def __init__(self, weight, getPredatorPosition, getPreyPosition, minDistance = 0):
        self.weight = weight
        self.getPredatorPosition = getPredatorPosition
        self.getPreyPosition = getPreyPosition
        self.minDistance = minDistance

    def __call__(self, state):
        predatorPos = self.getPredatorPosition(state)
        preyPos = self.getPreyPosition(state)

        distance = np.linalg.norm(predatorPos - preyPos, ord = 2)
        reward = -self.weight * max(0, distance - self.minDistance)

        return reward
