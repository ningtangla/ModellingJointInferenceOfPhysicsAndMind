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

class GetAgentPosFromState:
    def __init__(self, agentId, posIndex, numPosEachAgent):
        self.agentId = agentId
        self.posIndex = posIndex
        self.numPosEachAgent = numPosEachAgent

    def __call__(self, state):
        agentPos = state[self.agentId][self.posIndex:self.posIndex +
                                       self.numPosEachAgent]

        return agentPos

