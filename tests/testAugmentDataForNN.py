import sys
import os
src = os.path.join(os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
sys.path.append(os.path.join(os.pardir, 'exec'))
import unittest
from ddt import ddt, data, unpack
import numpy as np
import math
from evaluateAugmentation.augmentDataForNN import GenerateSymmetricData, \
    GenerateSymmetricState, GenerateSymmetricDistribution, CalibrateState
from analyticGeometryFunctions import transitePolarToCartesian
from dataTools import createSymmetricVector
xBoundary = [0, 180]
yBoundary = [0, 180]
xbias = xBoundary[1]
ybias = yBoundary[1]


@ddt
class TestGenerateData(unittest.TestCase):

    def setUp(self):
        sheepSpeed = 20
        degrees = [
            math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4,
            -math.pi * 3 / 4, -math.pi / 4, math.pi * 3 / 4
        ]
        self.sheepActionSpace = [
            tuple(np.round(sheepSpeed * transitePolarToCartesian(degree)))
            for degree in degrees
        ]
        self.symmetries = [
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([-1, 1])
        ]

    @data((2, 2, [1, 0, 0, 1], [1, 1], [0, 1, 1, 0]))
    @unpack
    def testGenerateSymmetricState(self, numOfAgent, stateDim, state, symmetry,
                                   groundTruth):
        round = lambda state: np.round(state, 10)
        calibrateState = CalibrateState(xBoundary, yBoundary, round)
        generateSymmetricState = GenerateSymmetricState(numOfAgent, stateDim,
                                                        createSymmetricVector,
                                                        calibrateState)
        testState = generateSymmetricState(state, np.array(symmetry))
        self.assertTrue(np.allclose(testState, np.array(groundTruth)))

    @data(([0.25, 0.3, 0, 0, 0.45, 0, 0,
            0], [1, 1], [0.3, 0.25, 0, 0, 0.45, 0, 0, 0]))
    @unpack
    def testGenerateSymmetricDistribution(self, distribution, symmetry,
                                          groundTruth):
        generateSymmetricDistribution = GenerateSymmetricDistribution(
            self.sheepActionSpace, createSymmetricVector)
        symmetricDistribution = generateSymmetricDistribution(
            distribution, np.array(symmetry))
        self.assertTrue(np.allclose(symmetricDistribution, groundTruth))

    # (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)
    @data((
        [[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0], [1]],
        2,
        2,
        [
            np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0]]),
            np.array([[5, 10, 5, 20], [0.3, 0.25, 0, 0, 0.45, 0, 0,
                                       0]]),  # symmetry: [1,1]
            np.array([[-10 + xbias, 5, -20 + xbias, 5],
                      [0.25, 0, 0.3, 0, 0, 0, 0, 0.45]]),  # symmetry: [0,1]
            np.array([[10, -5 + ybias, 20, -5 + ybias],
                      [0, 0.3, 0, 0.25, 0, 0, 0.45, 0]]),  # symmetry: [1,0]
            np.array([[-5 + xbias, -10 + ybias, -5 + xbias, -20 + ybias],
                      [0, 0, 0.25, 0.3, 0, 0.45, 0, 0]]),
            # symmetry: [-1,1]
            np.array([[-5 + xbias, 10, -5 + xbias, 20],
                      [0.3, 0, 0.25, 0, 0, 0, 0, 0.45]]),  # symmetry: [0,1]
            np.array([[5, -10 + ybias, 5, -20 + ybias],
                      [0, 0.25, 0, 0.3, 0, 0, 0.45, 0]]),  # symmetry: [1,0]
            np.array([[-10 + xbias, -5 + ybias, -20 + xbias, -5 + ybias],
                      [0, 0, 0.3, 0.25, 0, 0.45, 0, 0]])
        ]  # symmetry: [-1,1]
    ))
    @unpack
    def testGenerateSymmetricData(self, originalData, numOfAgent, stateDim,
                                  groundTruth):
        round = lambda state: np.round(state, 10)
        calibrateState = CalibrateState(xBoundary, yBoundary, round)
        generateSymmetricState = GenerateSymmetricState(numOfAgent, stateDim,
                                                        createSymmetricVector,
                                                        calibrateState)
        generateSymmetricDistribution = GenerateSymmetricDistribution(
            self.sheepActionSpace, createSymmetricVector)
        generateSymmetricData = GenerateSymmetricData(
            self.symmetries, self.sheepActionSpace, generateSymmetricState,
            generateSymmetricDistribution)

        symmetricDataSet = generateSymmetricData(originalData)
        symmetricDict = {
            tuple(np.round(data[0])): list(data[1]) for data in symmetricDataSet
        }
        groundTruthDict = {
            tuple(data[0]): list(data[1]) for data in groundTruth
        }
        for key in symmetricDict.keys():
            self.assertTrue(
                np.all(
                    np.array(symmetricDict[key]) == np.array(
                        groundTruthDict[key])))


if __name__ == "__main__":
    unittest.main()
