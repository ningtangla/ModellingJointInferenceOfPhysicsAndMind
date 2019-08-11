import numpy as np

class CalculateChasingSubtlety:
    def __init__(self, sheepId, wolfId, stateIndex, positionIndex):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex

    def __call__(self, traj):
        traj = np.array(traj)
        sheepVectorlist = [traj[i - 1][self.stateIndex][self.sheepId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]
        wolfVectorlist = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]

        sheepWolfAngleList = np.array([calculateIncludedAngle(v1, v2) for (v1, v2) in zip(sheepVectorlist, wolfVectorlist)])
        return sheepWolfAngleList


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


