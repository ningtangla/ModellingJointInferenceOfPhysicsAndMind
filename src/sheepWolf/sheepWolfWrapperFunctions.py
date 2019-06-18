import numpy as np
import pandas as pd
from functools import reduce


class GetAgentPosFromTrajectory:
    def __init__(self, timeStep, stateIndex, getAgentPosFromState):
        self.timeStep = timeStep
        self.stateIndex = stateIndex
        self.getAgentPosFromState = getAgentPosFromState

    def __call__(self, trajectory):
        stateAtTimeStep = trajectory[self.timeStep][self.stateIndex]
        posAtTimeStep = self.getAgentPosFromState(stateAtTimeStep)

        return posAtTimeStep


class GetTrialTrajectoryFromDf:
    def __init__(self, trialIndex):
        self.trialIndex = trialIndex

    def __call__(self, dataFrame):
        trajectory = dataFrame.values[self.trialIndex]
        return trajectory


class GetAgentPosFromState:
    def __init__(self, agentId, posIndex, numPosEachAgent):
        self.agentId = agentId
        self.posIndex = posIndex
        self.numPosEachAgent = numPosEachAgent
    def __call__(self, state):
        agentPos = state[self.agentId][self.posIndex:self.posIndex+self.numPosEachAgent]

        return agentPos


class GetAgentActionFromTrajectoryDf:
    def __init__(self, getTrialTrajectoryFromDf, timeStep, getAgentActionFromAllAgentActions, getAllAgentActionFromTrajectory):
        self.getTrialTrajectoryFromDf = getTrialTrajectoryFromDf
        self.timeStep = timeStep
        self.getAgentActionFromAllAgentActions = getAgentActionFromAllAgentActions
        self.getAllAgentActionFromTrajectory = getAllAgentActionFromTrajectory

    def __call__(self, trajectoryDf):
        trajectory = self.getTrialTrajectoryFromDf(trajectoryDf)
        allAgentActionsAtTimeStep = self.getAllAgentActionFromTrajectory(trajectory, self.timeStep)
        actionAtTimeStep = self.getAgentActionFromAllAgentActions(allAgentActionsAtTimeStep)
        actionSeries = pd.Series({'action': actionAtTimeStep})

        return actionSeries


class GetEpisodeLength:
    def __init__(self, getTrajectoryFromDf):
        self.getTrajectoryFromDf = getTrajectoryFromDf

    def __call__(self, trajectoryDf):
        trajectory = self.getTrajectoryFromDf(trajectoryDf)
        return len(trajectory)




