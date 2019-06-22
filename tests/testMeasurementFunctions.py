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
    @data(({"predict":np.array([0.228, 0.619, 0.153]), "target":np.array([0, 1, 0])}, 0.47965),
        ({"predict":np.array([0, 1, 0]), "target":np.array([0, 1, 0])}, 0))
    @unpack
    def testCrossEntropy(self, data, groundTruth):
        self.assertAlmostEqual(calculateCrossEntropy(data), groundTruth, places=5)


if __name__ == "__main__":
    unittest.main()