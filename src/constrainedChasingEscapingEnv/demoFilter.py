import numpy as np

class CalculateChasingDeviation:
    def __init__(self, sheepId, wolfId, stateIndex, positionIndex):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex

    def __call__(self, traj):
        traj = np.array(traj)
        heatSeekingVectorlist = [traj[i - 1][self.stateIndex][self.sheepId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]
        wolfVectorlist = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]

        sheepWolfAngleList = np.array([calculateIncludedAngle(v1, v2) for (v1, v2) in zip(heatSeekingVectorlist, wolfVectorlist)])
        return sheepWolfAngleList


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


class CalculateDistractorMoveDistance:
    def __init__(self, distractorId, stateIndex, positionIndex):
        self.distractorId = distractorId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex

    def __call__(self, traj):
        traj = np.array(traj)
        distractorVectorlist = [traj[i][self.stateIndex][self.distractorId][self.positionIndex] - traj[i - 1][self.stateIndex][self.distractorId][self.positionIndex] for i in range(1, len(traj))]
        distractorMoveDistances = [np.linalg.norm(distractorVector, ord = 2) for distractorVector in distractorVectorlist]
        return distractorMoveDistances

class OffsetMasterStates:
    def __init__(self, masterId, stateIndex, masterDelayStep):
        self.masterId = masterId
        self.stateIndex = stateIndex
        self.masterDelayStep = masterDelayStep

    def __call__(self, traj):
        traj = np.array(traj)
        masterStates = np.array([timeStep[self.stateIndex][self.masterId] for timeStep in traj[ : -self.masterDelayStep]])
        allAgentsStates = np.array([timeStep[self.stateIndex] for timeStep in traj[self.masterDelayStep: ]])
        allAgentsStates[:, self.masterId] = masterStates
        return allAgentsStates
