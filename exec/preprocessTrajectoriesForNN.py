import sys
sys.path.append("..")
import os
import numpy as np
import random
import functools as ft
import pickle
from exec.evaluationFunctions import GetSavePath


class AccumulateRewards:
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action, actionDist in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
        return accumulatedRewards


def addValuesToTraj(traj, trajValueFunc):
    values = trajValueFunc(traj)
    trajWithValues = [(s, a, dist, np.array([v])) for (s, a, dist), v in zip(traj, values)]
    return trajWithValues


def worldStatesToNpArrays(traj):
    newTraj = [(np.array(worldState).flatten(), action, actionDist, value) for worldState, action, actionDist, value in traj]
    return newTraj


def removeIrrelevantActions(traj, keepActionIndex):
    newTraj = [(state, worldAction[keepActionIndex], worldActionDist[keepActionIndex], value) for state, worldAction, worldActionDist, value in traj]
    return newTraj


def actionsToLabels(traj, actionSpace):
    actionToLabel = lambda action: np.array([1 if action == a else 0 for a in actionSpace])
    newTraj = [(state, actionToLabel(action), actionDist, value) for state, action, actionDist, value in traj]
    return newTraj


def actionDistsToProbs(traj, actionSpace):
    distToProbs = lambda dist: np.array([dist[a] for a in actionSpace])
    newTraj = [(state, action, distToProbs(actionDist), value) for state, action, actionDist, value in traj]
    return newTraj


def main():
    initPosition = np.array([[30, 30], [20, 20]])
    maxRollOutSteps = 10
    numSimulations = 200
    maxRunningSteps = 30
    numTrajs = 100
    useActionDist = True

    trajectoriesDir = '../data/trainingDataForNN/trajectories'
    extension = '.pickle'
    getTrajectoriesPath = GetSavePath(trajectoriesDir, extension)
    varDict = {}
    varDict["initPos"] = list(initPosition.flatten())
    varDict["rolloutSteps"] = maxRollOutSteps
    varDict["numSimulations"] = numSimulations
    varDict["maxRunningSteps"] = maxRunningSteps
    varDict["numTrajs"] = numTrajs
    varDict["withActionDist"] = useActionDist
    trajectoriesPath = getTrajectoriesPath(varDict)

    with open(trajectoriesPath, "rb") as f:
        trajs = pickle.load(f)

    trajs = [traj[:-1] for traj in trajs]

    decay = 1
    unstandardizedRewardFunc = lambda s, a: 1
    standardizedRewardFunc = lambda s, a: 1.0 / maxRunningSteps
    useStandardReward = False
    trajRewardFunc = AccumulateRewards(decay, unstandardizedRewardFunc if not useStandardReward else standardizedRewardFunc)
    trajsWithRewards = [addValuesToTraj(traj, trajRewardFunc) for traj in trajs]

    dataPoints = worldStatesToNpArrays(sum(trajsWithRewards, []))
    sheepID = 1
    dataWithSingleAgentActions = removeIrrelevantActions(dataPoints, sheepID)
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    dataWithLabels = actionsToLabels(dataWithSingleAgentActions, actionSpace)
    dataWithLabelsAndProbs = actionDistsToProbs(dataWithLabels, actionSpace)

    seed = 128
    random.seed(seed)
    random.shuffle(dataWithLabelsAndProbs)

    dataSetVarNames = ["state", "action", "actionDist", "value"]
    dataSetVarValues = [list(values) for values in zip(*dataWithLabelsAndProbs)]
    dataSet = dict(zip(dataSetVarNames, dataSetVarValues))

    dataSetsDir = '../data/trainingDataForNN/dataSets'
    if not os.path.exists(dataSetsDir):
        os.makedirs(dataSetsDir)
    getDataSetPath = GetSavePath(dataSetsDir, extension)
    varDict["standardizedReward"] = useStandardReward
    varDict["numDataPoints"] = len(dataWithLabelsAndProbs)
    dataSetPath = getDataSetPath(varDict)

    saveOn = True
    if saveOn:
        with open(dataSetPath, "wb") as f:
            pickle.dump(dataSet, f)
        print("Saved data set to {}".format(dataSetPath))


if __name__ == "__main__":
    main()