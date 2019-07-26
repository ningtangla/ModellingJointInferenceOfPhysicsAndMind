import numpy as np
import random


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

        trajectory = [self.currentState]

        for i in range(self.iterationNumber):
            allAgentNextAction = multiAgentPolicy(self.currentState)
            nextState = multiAgentTransition(allAgentNextAction, self.currentState)
            trajectory.append(nextState)

            self.currentState = nextState
            if self.isTerminal(self.currentState):
                break
        return trajectory


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory



class Sample3ObjectsTrajectory:
    def __init__(self, maxRunningSteps, transit, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class SampleTrajectoryTerminationProbability:
    def __init__(self, terminationProbability, transit, isTerminal, reset, chooseAction):
        self.terminationProbability = terminationProbability
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        terminal = False
        while(terminal == False):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState
            terminal = random.random() < self.terminationProbability

        return trajectory


def chooseGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def sampleAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedAction = actions[selectedIndex]
    return selectedAction


def getPairedTrajectory(agentsTrajectory):
    timeStepCount = len(agentsTrajectory[0])
    pairedTraj = [[agentTrajectory[timeStep] for agentTrajectory in agentsTrajectory] for timeStep in range(timeStepCount)]
    return pairedTraj
