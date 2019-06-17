import numpy as np
import pandas as pd

def computeDistance(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, optimalNextPosition, getPosAtNextStepFromTrajectory, getFirstTrajectoryFromDf):
        self.optimalNextPosition = optimalNextPosition
        self.getPosAtNextStepFromTrajectory = getPosAtNextStepFromTrajectory
        self.getFirstTrajectoryFromDf = getFirstTrajectoryFromDf

    def __call__(self, trajectoryDf):
        trajectory = self.getFirstTrajectoryFromDf(trajectoryDf)
        posAtNextStep = self.getPosAtNextStepFromTrajectory(trajectory)
        distance = computeDistance(self.optimalNextPosition, posAtNextStep)

        distanceSeries = pd.Series({'distance': distance})

        return distanceSeries