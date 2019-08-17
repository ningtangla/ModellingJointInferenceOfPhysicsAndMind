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
        distractorMoveDistances = [np.linalg.norm(distractorVector, ord=2) for distractorVector in distractorVectorlist]
        return distractorMoveDistances


class OffsetMasterStates:
    def __init__(self, masterId, stateIndex, masterDelayStep):
        self.masterId = masterId
        self.stateIndex = stateIndex
        self.masterDelayStep = masterDelayStep

    def __call__(self, traj):
        traj = np.array(traj)
        masterStates = np.array([np.array(timeStep)[self.stateIndex][self.masterId] for timeStep in traj[: -self.masterDelayStep]])
        allAgentsStates = np.array([timeStep[self.stateIndex] for timeStep in traj[self.masterDelayStep:]])
        allAgentsStates[:, self.masterId] = masterStates
        return allAgentsStates


class FindCirlceBetweenWolfAndMaster:
    def __init__(self, wolfId, masterId, stateIndex, positionIndex, timeWindow, angleDeviation):
        self.wolfId = wolfId
        self.masterId = masterId
        self.stateIndex = stateIndex
        self.positionIndex = positionIndex
        self.timeWindow = timeWindow
        self.angleDeviation = angleDeviation

    def __call__(self, traj):
        traj = np.array(traj)

        wolfVectorlist = [traj[i][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] for i in range(1, len(traj))]

        ropeVectorlist = [traj[i - 1][self.stateIndex][self.wolfId][self.positionIndex] - traj[i - 1][self.stateIndex][self.masterId][self.positionIndex] for i in range(1, len(traj))]

        ropeWolfAngleList = np.array([calculateIncludedAngle(v1, v2) for (v1, v2) in zip(wolfVectorlist, ropeVectorlist)])
        filterlist = [findCircleMove(ropeWolfAngleList[i:i + self.timeWindow], self.angleDeviation) for i in range(0, len(ropeWolfAngleList) - self.timeWindow + 1)]
        isValid = np.all(filterlist)
        return isValid


def findCircleMove(anglelist, angleVariance):
    lower = np.pi / 2 - angleVariance
    upper = np.pi / 2 + angleVariance
    filterAnglelist = list(filter(lambda x: x <= upper and x >= lower, anglelist))
    return len(filterAnglelist) < len(anglelist)
