import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import pandas as pd
import state
import envMujoco as env
import reward
from trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from collections import OrderedDict
import pickle


class RemoveNoiseFromState:

    def __init__(self, noiseIndex):
        self.noiseIndex = noiseIndex

    def __call__(self, trajectoryState):
        state = [np.delete(state, self.noiseIndex) for state in trajectoryState]
        return np.asarray(state).flatten()


class ProcessTrajectoryForNN:

    def __init__(self, agentId, actionToOneHot, removeNoiseFromState):
        self.agentId = agentId
        self.actionToOneHot = actionToOneHot
        self.removeNoiseFromState = removeNoiseFromState

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (self.removeNoiseFromState(state), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class ZeroValueInState:

    def __init__(self, valueIndex):
        self.valueIndex = valueIndex

    def __call__(self, state):
        stateArray = np.array(state)
        stateArray[self.valueIndex] = 0
        return stateArray


class AddFramesForTrajectory:

    def __init__(self, stateIndex, zeroValueInState):
        self.stateIndex = stateIndex
        self.zeroValueInState = zeroValueInState

    def __call__(self, trajectory, numOfFrame):
        augmentedTraj = list()
        for index in range(len(trajectory)):
            toCombinedStates = [
                trajectory[index - num + 1][self.stateIndex]
                if index - num + 1 > 0 else trajectory[0][self.stateIndex]
                for num in range(numOfFrame, 0, -1)
            ]
            if index < numOfFrame - 1:
                zeroedToCombinedStates = [
                    self.zeroValueInState(toCombinedStates[num])
                    if num < numOfFrame - index - 1 else toCombinedStates[num]
                    for num in range(numOfFrame)
                ]
                newState = np.concatenate(zeroedToCombinedStates)
            else:
                newState = np.concatenate(toCombinedStates)
            restInfo = trajectory[index][1:]
            newStep = [newState] + list(restInfo)
            augmentedTraj.append(newStep)
        return augmentedTraj


class PreProcessTrajectories:

    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                 processTrajectoryForNN, addFramesForTrajectory):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN
        self.addFramesForTrajectory = addFramesForTrajectory

    def __call__(self, trajectories, numOfFrame):
        trajectoriesWithValues = [
            self.addValuesToTrajectory(trajectory)
            for trajectory in trajectories
        ]
        filteredTrajectories = [
            self.removeTerminalTupleFromTrajectory(trajectory)
            for trajectory in trajectoriesWithValues
        ]
        processedTrajectories = [
            self.processTrajectoryForNN(trajectory)
            for trajectory in filteredTrajectories
        ]
        augmentedTrajectories = [
            self.addFramesForTrajectory(trajectory, numOfFrame)
            for trajectory in processedTrajectories
        ]
        allDataPoints = [
            dataPoint for trajectory in augmentedTrajectories
            for dataPoint in trajectory
        ]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]

        return trainData


class GenerateTrainingData:

    def __init__(self, getSavePathForData, preProcessTrajectories):
        self.getSavePathForData = getSavePathForData
        self.preProcessTrajectories = preProcessTrajectories

    def __call__(self, df, trajectories):
        numOfFrame = df.index.get_level_values('numOfFrame')[0]
        trainingData = self.preProcessTrajectories(trajectories, numOfFrame)
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        dataSavePath = self.getSavePathForData(parameters)
        with open(dataSavePath, 'wb') as f:
            pickle.dump(trainingData, f)
        return pd.Series({"dataSet": 'done'})


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data',
                           'evaluateByStateDimension')
    trajectoryDir = os.path.join(dataDir, "trainingTrajectories")
    trajectoryParameter = OrderedDict()
    trajectoryParameter['killzoneRadius'] = 2
    trajectoryParameter['maxRunningSteps'] = 25
    trajectoryParameter['numSimulations'] = 100
    trajectoryParameter['numTrajectories'] = 6000
    trajectoryParameter['qPosInitNoise'] = 9.7
    trajectoryParameter['qVelInitNoise'] = 8
    trajectoryParameter['rolloutHeuristicWeight'] = -0.1
    getTrajectorySavePath = GetSavePath(trajectoryDir, ".pickle",
                                        trajectoryParameter)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectories = loadTrajectories({})
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7),
                   (0, -10), (7, -7)]
    actionToOneHot = lambda action: np.asarray([
        1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
        for index in range(len(actionSpace))
    ])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][
        actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(
        getTerminalActionFromTrajectory)
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = state.GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = state.GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = 0.05
    playDeathBonus = -1
    playKillzoneRadius = 2
    playIsTerminal = env.IsTerminal(playKillzoneRadius, getWolfXPos,
                                    getSheepXPos)
    playReward = reward.RewardFunctionCompete(playAlivePenalty, playDeathBonus,
                                              playIsTerminal)
    qPosIndex = [0, 1]
    removeNoiseFromState = RemoveNoiseFromState(qPosIndex)
    processTrajectoryForNN = ProcessTrajectoryForNN(sheepId, actionToOneHot,
                                                    removeNoiseFromState)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)
    stateIndexInTrajectory = 0
    zeroIndex = [2, 3, 6, 7]
    zeroValueInState = ZeroValueInState(zeroIndex)
    addFramesForTrajectory = AddFramesForTrajectory(stateIndexInTrajectory,
                                                    zeroValueInState)
    preProcessTrajectories = PreProcessTrajectories(
        addValuesToTrajectory, removeTerminalTupleFromTrajectory,
        processTrajectoryForNN, addFramesForTrajectory)

    trainingDataDir = os.path.join(dataDir, "trainingData")
    if not os.path.exists(trainingDataDir):
        os.mkdir(trainingDataDir)
    getSavePathForData = GetSavePath(trainingDataDir, '.pickle')

    # split & apply
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['numOfFrame'] = [4]
    independentVariables['numOfStateSpace'] = [8]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    generateTrainingData = GenerateTrainingData(getSavePathForData,
                                                preProcessTrajectories)
    toSplitFrame.groupby(levelNames).apply(generateTrainingData, trajectories)


if __name__ == '__main__':
    main()
