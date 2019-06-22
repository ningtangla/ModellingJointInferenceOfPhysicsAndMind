import numpy as np

class GetAgentPosFromState:
    def __init__(self, agentId, posIndex):
        self.agentId = agentId
        self.posIndex = posIndex

    def __call__(self, state):
        state = np.asarray(state)
        agentPos = state[self.agentId][self.posIndex]

        return agentPos


class GetStateFromTrajectory:
    def __init__(self, timeStep, stateIndex):
        self.timeStep = timeStep
        self.stateIndex = stateIndex

    def __call__(self, trajectory):
        state = trajectory[self.timeStep][self.stateIndex]

        return state


class GetAgentPosFromTrajectory:
    def __init__(self, getAgentPosFromState, getStateFromTrajectory):
        self.getAgentPosFromState = getAgentPosFromState
        self.getStateFromTrajectory = getStateFromTrajectory

    def __call__(self, trajectory):
        state = self.getStateFromTrajectory(trajectory)
        agentPos = self.getAgentPosFromState(state)

        return agentPos


class GetAgentActionFromTrajectory:
    def __init__(self, timeStep, actionIndex, agentId):
        self.timeStep = timeStep
        self.actionIndex = actionIndex
        self.agentId = agentId

    def __call__(self, trajectory):
        allAgentActions = trajectory[self.timeStep][self.actionIndex]
        agentAction = allAgentActions[self.agentId]

        return agentAction