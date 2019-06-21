import numpy as np 
import pandas as pd 

class LocateAgent:
    def __init__(self, agentID, positionIndex):
        self.agentID = agentID
        self.positionIndex = positionIndex
    def __call__(self, state):
        agentState = state[self.agentID]
        agentPosition = [agentState[index] for index in self.positionIndex]
        return agentPosition


class MultiAgentSampleTrajectory:
    def __init__(self, agentNames, iterationNumber, isTerminal, reset, currentState = None):
        self.agentNames = agentNames
        self.iterationNumber = iterationNumber
        self.isTerminal = isTerminal
        self.reset = reset
        self.currentState = currentState


    def __call__(self, multiAgentPolicy, multiAgentTransition):
        if self.currentState is None:
            self.currentState = self.reset()

        locationDataFrame = pd.DataFrame([[agentState] for agentState in self.currentState], index = self.agentNames)
        for i in range(self.iterationNumber):
            allAgentNextAction = multiAgentPolicy(self.currentState)
            nextState = multiAgentTransition(allAgentNextAction, self.currentState)
            locationDataFrame[i+1] = nextState
            self.currentState = nextState
            if self.isTerminal(self.currentState):
                break
        return locationDataFrame

