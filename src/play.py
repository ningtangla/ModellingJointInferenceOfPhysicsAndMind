class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
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
