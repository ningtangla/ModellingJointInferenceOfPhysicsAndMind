import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
import math
import pickle
from analyticGeometryFunctions import transitePolarToCartesian
from dataTools import createSymmetricVector
from collections import OrderedDict
from trajectoriesSaveLoad import GetSavePath
from augmentDataForNN import GenerateSymmetricData, GenerateSymmetricDistribution, GenerateSymmetricState, CalibrateState


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data', 'augmentDataForNN')
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
    valueLabelKey = 'value'
    originalDataSet = [[state, label, value] for state, label, value in zip(
        dataSetDict[stateKey], dataSetDict[actionLabelKey], dataSetDict[valueLabelKey])]

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
    augmentedDataSetParameter['augmented'] = True
    augmentedDataSetPath = getSavePathForDataSet(augmentedDataSetParameter)
    augmentedDataSetDict = dataSetDict
    augmentedDataSetDict[stateKey] = [
        state for state, label, value in augmentedDataSet
    ]
    augmentedDataSetDict[actionLabelKey] = [
        label for state, label, value in augmentedDataSet
    ]
    print(len(augmentedDataSetDict[stateKey]))
    with open(augmentedDataSetPath, 'wb') as f:
        pickle.dump(augmentedDataSet, f)


if __name__ == "__main__":
    main()