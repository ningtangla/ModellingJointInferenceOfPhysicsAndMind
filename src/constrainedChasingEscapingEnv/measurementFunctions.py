import numpy as np


def computeDistance(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, optimalNextPosition, getPosAtNextStepFromTrajectory):
        self.optimalNextPosition = optimalNextPosition
        self.getPosAtNextStepFromTrajectory = getPosAtNextStepFromTrajectory

    def __call__(self, trajectory):
        posAtNextStep = self.getPosAtNextStepFromTrajectory(trajectory)
        distance = computeDistance(self.optimalNextPosition, posAtNextStep)

        return distance

def calculateCrossEntropy(data, episilon = 1e-12):
    prediction, target = list(data.values())
    ce = -1 * sum([target[index] * np.log(prediction[index]+episilon) for index in range(len(prediction))])
    return ce
