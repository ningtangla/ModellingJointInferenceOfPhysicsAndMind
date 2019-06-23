import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os
sys.path.append(os.path.join('..', 'src', 'constrainedChasingEscapingEnv'))

from analyticGeometryFunctions import transiteCartesianToPolar, transitePolarToCartesian, computeAngleBetweenVectors, computeVectorNorm 


@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
    @data((np.array([0, 0]), 0), (np.array([1, 1]), 0.785))
    @unpack
    def testTransiteCartesianToPolar(self, vector, groundTruthAngle):
        returnedValue = transiteCartesianToPolar(vector)
        self.assertAlmostEqual(returnedValue, groundTruthAngle, places=3)

    @data((0, np.array([1, 0])), (np.pi / 4, np.array([0.707, 0.707])))
    @unpack
    def testTransitePolarToCartesian(self, angle, groundTruthCoordinates):
        returnedValue = transitePolarToCartesian(angle)
        self.assertAlmostEqual(returnedValue.all(), groundTruthCoordinates.all(), places=3)

    @data((np.array([1, 1]), np.array([1, 1]), 0), (np.array([1, 0]), np.array([0, 1]), 1.571))
    @unpack
    def testComputeAngleBetweenVectors(self, vector1, vector2, groundTruthAngle):
        returnedValue = computeAngleBetweenVectors(vector1, vector2)
        self.assertAlmostEqual(returnedValue, groundTruthAngle, places=3)

    @data((np.asarray([1, 2]), np.sqrt(5)),
          (np.asarray([-10, 10]), 10*np.sqrt(2)))
    @unpack
    def testComputeDistance(self, vector, groundTruthDistance):
        norm = computeVectorNorm(vector)

        self.assertEqual(norm, groundTruthDistance)


if __name__ == "__main__":
    unittest.main()
