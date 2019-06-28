import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import numpy as np
import math
import pickle
import os
from analyticGeometryFunctions import transitePolarToCartesian
from dataTools import createSymmetricVector


class GenerateSymmetricData():
    def __init__(self, bias, getSymmeticVector, symmetries, actionSpace):
        self.bias = bias
        self.getSymmetricVector = getSymmeticVector
        self.symmetries = symmetries
        self.actionSpace = actionSpace

    def __call__(self, dataSet):
        newDataSet = []
        for data in dataSet:
            turningPoint = None
            state, actionDistribution = data
            for symmetry in self.symmetries:
                newState = np.concatenate(
                    [self.getSymmetricVector(symmetry, np.array(state[0:2])),
                     self.getSymmetricVector(symmetry, np.array(state[2:4]))])
                newActionDistributionDict = {tuple(np.round(
                    self.getSymmetricVector(symmetry, np.array(self.actionSpace[index])))):
                                                 actionDistribution[index] for index
                                             in range(len(actionDistribution))}
                newActionDistribution = [newActionDistributionDict[action] for
                                         action in self.actionSpace]
                if np.all(symmetry == np.array([1, 1])):
                    turningPoint = np.array(
                        [newState, newActionDistribution])
                newDataSet.append(
                    np.array([newState, np.array(newActionDistribution)]))
            if turningPoint is None:
                continue
            state, actionDistribution = turningPoint
            for symmetry in self.symmetries:
                newState = np.concatenate(
                    [self.getSymmetricVector(symmetry, np.array(state[0:2])),
                     self.getSymmetricVector(symmetry, np.array(state[2:4]))])
                newActionDistributionDict = {tuple(np.round(
                    self.getSymmetricVector(symmetry, np.array(self.actionSpace[index])))):
                                                 actionDistribution[index] for index
                                             in range(len(actionDistribution))}
                newActionDistribution = [newActionDistributionDict[action] for
                                         action in self.actionSpace]
                newDataSet.append(
                    np.array([newState, np.array(newActionDistribution)]))
        augmentedDataSet = [np.array(
            [np.array([coor + self.bias if coor < 0 else coor for coor in state]),
             distribution]) for (state, distribution) in newDataSet]
        return augmentedDataSet


def main():
    stateKey = 'state'
    actionLabelKey = 'actionDist'
    dataDir = '../data/augmentedDataForNN'
    originalDataSetName = "cBase=100_initPos=Random_maxRunningSteps=30_numDataPoints=68181_numSimulations=200_numTrajs=2500_rolloutSteps=10_standardizedReward=True.pickle"
    dataSetPath = os.path.join(dataDir, "dataSets", originalDataSetName)
    if not os.path.exists(dataSetPath):
        print("No dataSet in:\n{}".format(dataSetPath))
        exit(1)
    with open(dataSetPath, "rb") as f:
        originalDataSet = pickle.load(f)
    modifiedDataSet = [[state, label]for state, label in zip(originalDataSet[stateKey], originalDataSet[actionLabelKey])]

    # env
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    bias = xBoundary[1]
    sheepSpeed = 20
    degrees = [math.pi / 2, 0, math.pi, -math.pi / 2,
               math.pi / 4, -math.pi * 3 / 4, -math.pi / 4, math.pi * 3 / 4]
    sheepActionSpace = [
        tuple(np.round(sheepSpeed * transitePolarToCartesian(degree))) for
        degree in degrees]
    symmetries = [np.array([1, 1]), np.array([0, 1]), np.array([1, 0]),
                  np.array([-1, 1])]
    generateSymmetricData = GenerateSymmetricData(bias, createSymmetricVector,
                                                  symmetries, sheepActionSpace)
    augmentedDataSet = generateSymmetricData(modifiedDataSet)
    augmentedDataSetDir = os.path.join(dataDir, "augmentedDataSets")
    if not os.path.exists(augmentedDataSetDir):
        os.mkdir(augmentedDataSetDir)
    augmentedDataSetPath = os.path.join(augmentedDataSetDir, originalDataSetName)
    originalDataSet[stateKey] = [state for state, label in augmentedDataSet]
    originalDataSet[actionLabelKey] = [label for state, label in augmentedDataSet]
    with open(augmentedDataSetPath, 'wb') as f:
        pickle.dump(originalDataSet, f)



if __name__ == "__main__":
    main()