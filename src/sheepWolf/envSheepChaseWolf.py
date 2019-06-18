import numpy as np
import pandas as pd
from functools import reduce


def stationaryWolfPolicy(worldState):
    return (0, 0)


class WolfPolicyForceDirectlyTowardsSheep:
    def __init__(self, getSheepXPos, getWolfXPos, wolfActionMagnitude):
        self.getSheepXPos = getSheepXPos
        self.getWolfXPos = getWolfXPos
        self.wolfActionMagnitude = wolfActionMagnitude

    def __call__(self, worldState):
        sheepXPos = self.getSheepXPos(worldState)
        wolfXPos = self.getWolfXPos(worldState)

        sheepAction = sheepXPos - wolfXPos
        sheepActionNorm = np.sum(np.abs(sheepAction))
        if sheepActionNorm != 0:
            sheepAction = sheepAction / sheepActionNorm
            sheepAction *= self.wolfActionMagnitude

        return sheepAction


def computeDistance(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


class GetAgentPosFromTrajectory:
    def __init__(self, timeStep, stateIndex, agentId, posIndex, numPosEachAgent):
        self.timeStep = timeStep
        self.stateIndex = stateIndex
        self.agentId = agentId
        self.posIndex = posIndex
        self.numPosEachAgent = numPosEachAgent

    def __call__(self, trajectory):
        stateAtTimeStep = trajectory[self.timeStep][self.stateIndex]
        posAtTimeStep = stateAtTimeStep[self.agentId][self.posIndex:
                                                      self.posIndex + self.numPosEachAgent]

        return posAtTimeStep


class GetTrialTrajectoryFromDf:
    def __init__(self, trialIndex):
        self.trialIndex = trialIndex

    def __call__(self, dataFrame):
        trajectory = dataFrame.values[self.trialIndex]
        return trajectory


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, optimalNextPosition, getPosAtNextStepFromTrajectory, getFirstTrajectoryFromDf):
        self.optimalNextPosition = optimalNextPosition
        self.getPosAtNextStepFromTrajectory = getPosAtNextStepFromTrajectory
        self.getFirstTrajectoryFromDf = getFirstTrajectoryFromDf

    def __call__(self, trajectoryDf):
        trajectory = self.getFirstTrajectoryFromDf(trajectoryDf)
        posAtNextStep = self.getPosAtNextStepFromTrajectory(trajectory)
        distance = computeDistance(self.optimalNextPosition, posAtNextStep)

        distanceSeries = pd.Series({'distance': distance})

        return distanceSeries


class GetAgentPos:
    def __init__(self, agentId, posIndex, numPosEachAgent):
        self.agentId = agentId
        self.posIndex = posIndex
        self.numPosEachAgent = numPosEachAgent

    def __call__(self, state):
        agentPos = state[self.agentId][self.posIndex:self.posIndex +
                                       self.numPosEachAgent]

        return agentPos


class GetAgentActionFromTrajectoryDf:
    def __init__(self, getTrialTrajectoryFromDf, timeStep, getAgentActionFromAllAgentActions, getAllAgentActionFromTrajectory):
        self.getTrialTrajectoryFromDf = getTrialTrajectoryFromDf
        self.timeStep = timeStep
        self.getAgentActionFromAllAgentActions = getAgentActionFromAllAgentActions
        self.getAllAgentActionFromTrajectory = getAllAgentActionFromTrajectory

    def __call__(self, trajectoryDf):
        trajectory = self.getTrialTrajectoryFromDf(trajectoryDf)
        allAgentActionsAtTimeStep = self.getAllAgentActionFromTrajectory(
            trajectory, self.timeStep)
        actionAtTimeStep = self.getAgentActionFromAllAgentActions(
            allAgentActionsAtTimeStep)
        actionSeries = pd.Series({'action': actionAtTimeStep})

        return actionSeries


class GetEpisodeLength:
    def __init__(self, getTrajectoryFromDf):
        self.getTrajectoryFromDf = getTrajectoryFromDf

    def __call__(self, trajectoryDf):
        trajectory = self.getTrajectoryFromDf(trajectoryDf)
        return len(trajectory)


def stationaryAgentPolicy(state):
    return (0, 0)


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionIndex = np.random.randint(len(self.actionSpace))
        action = self.actionSpace[actionIndex]
        return action


class HeatSeekingPolicy:
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


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
    else:
        innerProduct = vectoriseInnerProduct
    angle = np.arccos(
        innerProduct / (computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle


def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(axis=0), 0.5)
