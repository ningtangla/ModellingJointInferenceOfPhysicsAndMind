import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import unittest
from ddt import ddt, data, unpack
import numpy as np
from analyticGeometryFunctions import createActionVector

@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):

	@data((8, 2, [(2, 0), (np.power(2,0.5), np.power(2,0.5)), (0, 2), (-np.power(2,0.5), np.power(2,0.5)), (-2, 0), (-np.power(2,0.5), -np.power(2,0.5)),
	              (0, -2), (np.power(2,0.5), -np.power(2,0.5))]))
	@unpack
	def testCreateActionVector(self, discreteFactor, magnitude, groundTruth):
		arrayGroundTruth = [np.array(action) for action in groundTruth]
		self.assertTrue(np.allclose(createActionVector(discreteFactor, magnitude), np.array(arrayGroundTruth)))

if __name__ == "__main__":
	unittest.main()