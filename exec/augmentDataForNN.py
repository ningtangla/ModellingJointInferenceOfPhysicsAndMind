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


class CalibrateState:

    def __init__(self, xBoundary, yBoundary, round):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.round = round

    def __call__(self, state):
        xPos, yPos = self.round(state)
        if xPos < self.xBoundary[0]:
            calibrateXPos = xPos + self.xBoundary[1]
        else:
            calibrateXPos = xPos
        if yPos < self.yBoundary[0]:
            calibrateyPos = yPos + self.yBoundary[1]
        else:
            calibrateyPos = yPos
        return [calibrateXPos, calibrateyPos]


class GenerateSymmetricState:

    def __init__(self, numOfAgent, stateDim, createSymmetricVector,
                 calibrateState):
        self.numOfAgent = numOfAgent
        self.stateDim = stateDim
        self.createSymmetricVector = createSymmetricVector
        self.calibrateState = calibrateState

    def __call__(self, state, symmetry):
        groupedState = [
            state[num * self.stateDim:(num + 1) * self.stateDim]
            for num in range(self.numOfAgent)
        ]
        symmetricState = [
            self.calibrateState(createSymmetricVector(symmetry, state))
            for state in groupedState
        ]
        flattenedState = np.concatenate(symmetricState)
        return flattenedState


class GenerateSymmetricDistribution:

    def __init__(self, actionSpace, createSymmetricVector):
        self.createSymmetricVector = createSymmetricVector
        self.actionSpace = actionSpace

    def __call__(self, distribution, symmetry):
        distributionDict = {
            action: prob for action, prob in zip(self.actionSpace, distribution)
        }
        getSymmetricAction = lambda action: tuple(
            np.around(self.createSymmetricVector(symmetry, action)))
        symmetricDistribution = [
            distributionDict[getSymmetricAction(np.array(action))]
            for action in self.actionSpace
        ]
        return symmetricDistribution


class GenerateSymmetricData:

    def __init__(self, symmetries, actionSpace, generateSymmetricState,
                 generateSymmetricDistribution):
        self.symmetries = symmetries
        self.actionSpace = actionSpace
        self.generateSymmetricState = generateSymmetricState
        self.generateSymmetricDistribution = generateSymmetricDistribution

    def __call__(self, data):
        state, distribution = data
        twinSymmetry = self.symmetries[0]
        twinState = self.generateSymmetricState(state, twinSymmetry)
        twinDistribution = self.generateSymmetricDistribution(
            distribution, twinSymmetry)
        symmetricData = {
            tuple(self.generateSymmetricState(state, symmetry)):
            self.generateSymmetricDistribution(distribution, symmetry)
            for symmetry in self.symmetries
        }
        twinSymmetricData = {
            tuple(self.generateSymmetricState(twinState, symmetry)):
            self.generateSymmetricDistribution(twinDistribution, symmetry)
            for symmetry in self.symmetries
        }
        symmetricData.update(twinSymmetricData)
        augmentedData = [[state, distribution]
                         for state, distribution in symmetricData.items()]
        return augmentedData


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
        dataSetDict = pickle.load(f)
    stateKey = 'state'
    actionLabelKey = 'actionDist'
    originalDataSet = [[state, label] for state, label in zip(
        dataSetDict[stateKey], dataSetDict[actionLabelKey])]

    # env
    xBoundary = [0, 180]
    yBoundary = [0, 180]
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
    stateDim = 2
    numOfAgent = 2
    round = lambda state: np.round(state, 10)
    calibrateState = CalibrateState(xBoundary, yBoundary, round)
    generateSymmetricState = GenerateSymmetricState(numOfAgent, stateDim,
                                                    createSymmetricVector,
                                                    calibrateState)
    generateSymmetricDistribution = GenerateSymmetricDistribution(
        sheepActionSpace, createSymmetricVector)
    generateSymmetricData = GenerateSymmetricData(
        symmetries, createSymmetricVector, generateSymmetricState,
        generateSymmetricDistribution)
    augmentedDataSet = np.concatenate([
        generateSymmetricData(originalData) for originalData in originalDataSet
    ])
    augmentedDataSetParameter = dataSetParameter
    augmentedDataSetParameter['augmented'] = 'yes'
    augmentedDataSetPath = getSavePathForDataSet(augmentedDataSetParameter)
    augmentedDataSetDict = dataSetDict
    augmentedDataSetDict[stateKey] = [
        state for state, label in augmentedDataSet
    ]
    augmentedDataSetDict[actionLabelKey] = [
        label for state, label in augmentedDataSet
    ]
    with open(augmentedDataSetPath, 'wb') as f:
        pickle.dump(augmentedDataSet, f)


if __name__ == "__main__":
    main()
