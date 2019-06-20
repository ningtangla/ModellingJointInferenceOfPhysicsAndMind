import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import unittest
from ddt import ddt, data, unpack
import numpy as np
from dataTools import createSymmetricVector

@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
    @data((np.array([1, 1]), np.array([0.5, 0]), np.array([0, 0.5])))
    @unpack
    def testcreateSymmetricVector(self, symmetricAxis, originalVector, groundTruth):
        self.assertTrue(np.allclose(createSymmetricVector(symmetricAxis, originalVector), groundTruth, rtol=1e-05, atol=1e-08))


if __name__ == "__main__":
    unittest.main()