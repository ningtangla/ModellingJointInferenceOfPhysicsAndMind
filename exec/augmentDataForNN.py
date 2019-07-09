import sys
import os
src = os.path.join(os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import math
import pickle
from analyticGeometryFunctions import transitePolarToCartesian
from dataTools import createSymmetricVector
from collections import OrderedDict
from evaluationFunctions import GetSavePath


class GetAgentStateFromDataSetState:

    def __init__(self, agentStateDim, agentID):
        self.agentStateDim = agentStateDim
        self.agentID = agentID

    def __call__(self, dataSetState):
        indexes = [
            self.agentID * self.agentStateDim + num
            for num in range(self.agentStateDim)
        ]
        agentState = [dataSetState[index] for index in indexes]
        return agentState


class GenerateSymmetricData():

    def __init__(self, bias, getSymmeticVector, symmetries, actionSpace,
                 getSheepState, getWolfState):
        self.bias = bias
        self.getSymmetricVector = getSymmeticVector
        self.symmetries = symmetries
        self.actionSpace = actionSpace
        self.getSheepState = getSheepState
        self.getWolfState = getWolfState

    def __call__(self, dataSet):
        newDataSet = []
        for data in dataSet:
            turningPoint = None
            state, actionDistribution = data
            for symmetry in self.symmetries:
                newState = np.concatenate([
                    self.getSymmetricVector(symmetry,
                                            np.array(
                                                self.getSheepState(state))),
                    self.getSymmetricVector(symmetry,
                                            np.array(self.getWolfState(state)))
                ])
                newActionDistributionDict = {
                    tuple(
                        np.round(
                            self.getSymmetricVector(
                                symmetry, np.array(self.actionSpace[index])))):
                    actionDistribution[index]
                    for index in range(len(actionDistribution))
                }
                newActionDistribution = [
                    newActionDistributionDict[action]
                    for action in self.actionSpace
                ]
                if np.all(symmetry == np.array([1, 1])):
                    turningPoint = np.array([newState, newActionDistribution])
                newDataSet.append(
                    np.array([newState,
                              np.array(newActionDistribution)]))
            if turningPoint is None:
                continue
            state, actionDistribution = turningPoint
            for symmetry in self.symmetries:
                newState = np.concatenate([
                    self.getSymmetricVector(symmetry,
                                            np.array(
                                                self.getSheepState(state))),
                    self.getSymmetricVector(symmetry,
                                            np.array(self.getWolfState(state)))
                ])
                newActionDistributionDict = {
                    tuple(
                        np.round(
                            self.getSymmetricVector(
                                symmetry, np.array(self.actionSpace[index])))):
                    actionDistribution[index]
                    for index in range(len(actionDistribution))
                }
                newActionDistribution = [
                    newActionDistributionDict[action]
                    for action in self.actionSpace
                ]
                newDataSet.append(
                    np.array([newState,
                              np.array(newActionDistribution)]))
        augmentedDataSet = [
            np.array([
                np.array(
                    [coor + self.bias if coor < 0 else coor
                     for coor in state]), distribution
            ])
            for (state, distribution) in newDataSet
        ]
        return augmentedDataSet


def main():
    dataDir = os.path.join(os.pardir, 'data', 'augmentDataForNN')
    dataSetParameter = OrderedDict()
    dataSetParameter['cBase'] = 100
    dataSetParameter['initPos'] = 'Random'
    dataSetParameter['maxRunningSteps'] = 30
    dataSetParameter['numDataPoints'] = 68181
    dataSetParameter['numSimulations'] = 200
    dataSetParameter['numTrajs'] = 2500
    dataSetParameter['rolloutSteps'] = 10
    dataSetParameter['standardizedReward'] = 'True'
    dataSetExtension = '.pickle'
    getSavePathForDataSet = GetSavePath(dataDir, dataSetExtension)
    dataSetPath = getSavePathForDataSet(dataSetParameter)
    if not os.path.exists(dataSetPath):
        print("No dataSet in: {}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        dataSet = pickle.load(f)
    stateKey = 'state'
    actionLabelKey = 'actionDist'
    originalData = [[
        state, label
    ] for state, label in zip(dataSet[stateKey], dataSet[actionLabelKey])]

    # env
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    bias = xBoundary[1]
    degrees = [
        math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4, -math.pi * 3 / 4,
        -math.pi / 4, math.pi * 3 / 4
    ]
    establishAction = lambda speed, degree: tuple(
        np.round(speed * transitePolarToCartesian(degree)))
    sheepSpeed = 20
    sheepActionSpace = [
        establishAction(sheepSpeed, degree) for degree in degrees
    ]
    symmetries = [
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([-1, 1])
    ]
    agentStateDim = 2
    sheepID = 0
    getSheepState = GetAgentStateFromDataSetState(agentStateDim, sheepID)
    wolfID = 1
    getWolfState = GetAgentStateFromDataSetState(agentStateDim, wolfID)
    generateSymmetricData = GenerateSymmetricData(bias, createSymmetricVector,
                                                  symmetries, sheepActionSpace,
                                                  getSheepState, getWolfState)
    augmentedData = generateSymmetricData(originalData)
    augmentedDataSetParameter = dataSetParameter
    augmentedDataSetParameter['augmented'] = 'yes'
    augmentedDataSetPath = getSavePathForDataSet(augmentedDataSetParameter)
    augmentedDataSet = dataSet
    augmentedDataSet[stateKey] = [state for state, label in augmentedData]
    augmentedDataSet[actionLabelKey] = [label for state, label in augmentedData]
    with open(augmentedDataSetPath, 'wb') as f:
        pickle.dump(augmentedDataSet, f)


if __name__ == "__main__":
    main()
