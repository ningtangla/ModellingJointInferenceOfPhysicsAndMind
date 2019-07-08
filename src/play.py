import numpy as np
import pandas as pd

def computeDistanceFromState(state):
    pos1 = np.asarray(state)[0][2:4]
    pos2 = np.asarray(state)[1][2:4]

    return np.sqrt(np.sum(np.square(pos1 - pos2)))


class MultiAgentSampleTrajectory:
    def __init__(self, agentNames, iterationNumber, isTerminal, reset, currentState=None):
        self.agentNames = agentNames
        self.iterationNumber = iterationNumber
        self.isTerminal = isTerminal
        self.reset = reset
        self.currentState = currentState

    def __call__(self, multiAgentPolicy, multiAgentTransition):
        if self.currentState is None:
            self.currentState = self.reset()

        locationDataFrame = pd.DataFrame([[agentState] for agentState in self.currentState], index=self.agentNames)
        for i in range(self.iterationNumber):
            allAgentNextAction = multiAgentPolicy(self.currentState)
            nextState = multiAgentTransition(allAgentNextAction, self.currentState)
            locationDataFrame[i + 1] = nextState
            self.currentState = nextState
            if self.isTerminal(self.currentState):
                break
        return locationDataFrame

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, maxInitDistance):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.maxInitDistance = maxInitDistance

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state) or computeDistanceFromState(state) > self.maxInitDistance:
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None))
                break
            action = policy(state)
            trajectory.append((state, action))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class SampleTrajectoryWithActionDist:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, distToAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.distToAction = distToAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDist = policy(state)
            action = self.distToAction(actionDist)
            trajectory.append((state, action, actionDist))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


def agentDistToGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def worldDistToAction(agentDistToAction, worldDist):
    worldAction = [agentDistToAction(dist) if type(dist) is dict else dist for dist in worldDist]
    return worldAction
