import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import unittest
from ddt import ddt, data, unpack
import numpy as np
from measurementFunctions import calculateCrossEntropy

@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
    @data((np.array([0.228, 0.619, 0.153]), np.array([0, 1, 0]), 0.47965))
    @unpack
    def testCrossEntropy(self, prediction, target, groundTruth):
        self.assertAlmostEqual(calculateCrossEntropy(prediction, target), groundTruth, places=5)


if __name__ == "__main__":
    unittest.main()