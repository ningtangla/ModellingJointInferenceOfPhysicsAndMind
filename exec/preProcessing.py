import numpy as np
from functools import reduce

class AccumulateRewards:
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        try:
            rewards = [self.rewardFunction(state, action) for state, action, actionDist in trajectory]
        except:
            import ipdb; ipdb.set_trace()
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])

        return accumulatedRewards


class AddValuesToTrajectory:
    def __init__(self, trajectoryValueFunction):
        self.trajectoryValueFunction = trajectoryValueFunction

    def __call__(self, trajectory):
        values = self.trajectoryValueFunction(trajectory)
        trajWithValues = [(s, a, dist, np.array([v])) for (s, a, dist), v in zip(trajectory, values)]

        return trajWithValues


class RemoveTerminalTupleFromTrajectory:
    def __init__(self, getTerminalActionFromTrajectory):
        self.getTerminalActionFromTrajectory = getTerminalActionFromTrajectory

    def __call__(self, trajectory):
        terminalAction = self.getTerminalActionFromTrajectory(trajectory)
        if terminalAction is None:
            return trajectory[:-1]
        else:
            return trajectory






