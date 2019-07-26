import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
import numpy as np
from dataTools import createSymmetricVector

class CalibrateState:

    def __init__(self, xBoundary, yBoundary, round):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.round = round

    def __call__(self, state):
        xPos, yPos = self.round(state)
        xbias = self.xBoundary[1] - self.xBoundary[0]
        ybias = self.yBoundary[1] - self.yBoundary[0]
        if xPos < self.xBoundary[0]:
            calibrateXPos = xPos + xbias
        elif xPos > self.xBoundary[1]:
            calibrateXPos = xPos - xbias
        else:
            calibrateXPos = xPos
        if yPos < self.yBoundary[0]:
            calibrateyPos = yPos + ybias
        elif yPos > self.yBoundary[1]:
            calibrateyPos = yPos - ybias
        else:
            calibrateyPos = yPos
        return [calibrateXPos, calibrateyPos]


class GenerateSymmetricState:

    def __init__(self, numOfAgent, stateDim, xPosIndex, velIndex,
                 createSymmetricVector, calibrateState):
        self.numOfAgent = numOfAgent
        self.stateDim = stateDim
        self.xPosIndex = xPosIndex
        self.velIndex = velIndex
        self.createSymmetricVector = createSymmetricVector
        self.calibrateState = calibrateState

    def __call__(self, state, symmetry):
        groupedState = [
            np.array(state[num * self.stateDim:(num + 1) * self.stateDim])
            for num in range(self.numOfAgent)
        ]
        symmetricState = []
        for state in groupedState:
            state[self.xPosIndex] = self.calibrateState(createSymmetricVector(symmetry, state[self.xPosIndex]))
            state[self.velIndex] = self.calibrateState(createSymmetricVector(symmetry, state[self.velIndex]))
            symmetricState.append(state)
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

    def __init__(self, symmetries, generateSymmetricState,
                 generateSymmetricDistribution):
        self.symmetries = symmetries
        self.generateSymmetricState = generateSymmetricState
        self.generateSymmetricDistribution = generateSymmetricDistribution

    def __call__(self, data):
        state, distribution, value = data
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
        augmentedData = [[state, distribution, value]
                         for state, distribution in symmetricData.items()]
        return augmentedData
