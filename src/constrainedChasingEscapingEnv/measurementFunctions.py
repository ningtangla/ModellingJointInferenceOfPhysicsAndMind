import numpy as np


class ComputeOptimalNextPos:
    def __init__(self, getInitStateFromTrajectory, optimalPolicy, transit, getAgentPosFromState):
        self.getInitStateFromTrajectory = getInitStateFromTrajectory
        self.optimalPolicy = optimalPolicy
        self.transit = transit
        self.getAgentPosFromState = getAgentPosFromState

    def __call__(self, trajectory):
        initState = self.getInitStateFromTrajectory(trajectory)
        optimalAction = self.optimalPolicy(initState)
        nextState = self.transit(initState, optimalAction)
        nextPos = self.getAgentPosFromState(nextState)

        return nextPos


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, computeOptimalNextPos, getPosAtNextStepFromTrajectory):
        self.computeOptimalNextPos = computeOptimalNextPos
        self.getPosAtNextStepFromTrajectory = getPosAtNextStepFromTrajectory

    def __call__(self, trajectory):
        optimalNextPos = self.computeOptimalNextPos(trajectory)
        posAtNextStep = self.getPosAtNextStepFromTrajectory(trajectory)
        L2distance = np.linalg.norm(posAtNextStep - optimalNextPos, ord = 2)

        return L2distance

def calculateCrossEntropy(prediction, target, episilon = 1e-12):
    ce = -1 * sum([target[index] * np.log(prediction[index]+episilon) for index in range(len(prediction))])
    return ce
