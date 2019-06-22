import numpy as np


def computeDistance(pos1, pos2):
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


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
        distance = computeDistance(posAtNextStep, optimalNextPos)

        return distance


def computeVectorNorm(vector):
    return np.linalg.norm(vector, 2, 0)


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
    else:
        innerProduct = vectoriseInnerProduct
    angle = np.arccos(innerProduct/(computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle