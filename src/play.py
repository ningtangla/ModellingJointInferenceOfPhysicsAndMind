import numpy as np

def computeDistanceFromState(state):
    pos1 = np.asarray(state)[0][2:4]
    pos2 = np.asarray(state)[1][2:4]

    return np.sqrt(np.sum(np.square(pos1 - pos2)))

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
