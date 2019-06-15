import unittest
from ddt import ddt, data, unpack
import numpy as np
from AnalyticGeometryFunctions import calculateCrossEntropy, getSymmetricVector

@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
	@data((np.array([0.228, 0.619, 0.153]), np.array([0, 1, 0]), 0.47965))
	@unpack
	def testCrossEntropy(self, prediction, target, groundTruth):
		self.assertAlmostEqual(calculateCrossEntropy(prediction, target), groundTruth, places=5)

	@data((np.array([1, 1]), np.array([0.5, 0]), np.array([0, 0.5])))
	@unpack
	def testgetSymmetricVector(self, symmetricAxis, originalVector, groundTruth):
		self.assertTrue(np.allclose(getSymmetricVector(symmetricAxis, originalVector), groundTruth, rtol=1e-05, atol=1e-08))

if __name__ == "__main__":
	unittest.main()