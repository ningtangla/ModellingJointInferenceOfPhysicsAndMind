import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../exec")
sys.path.append("../src")
import unittest
from ddt import ddt, data, unpack
import numpy as np
import math
from augmentDataForNN import GenerateSymmetricData
from analyticGeometryFunctions import transitePolarToCartesian
from dataTools import createSymmetricVector
xBoundary = [0, 180]
yBoundary = [0, 180]
xbias = xBoundary[1]
ybias = yBoundary[1]

@ddt
class TestGenerateData(unittest.TestCase):
	# (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)
	@data(([np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0]])],
		   [np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0]]),
		   np.array([[5, 10, 5, 20], [0.3, 0.25, 0, 0, 0.45, 0, 0, 0]]), # symmetry: [1,1]
           np.array([[-10+xbias, 5, -20+xbias, 5], [0.25, 0, 0.3, 0, 0, 0, 0, 0.45]]),  # symmetry: [0,1]
           np.array([[10, -5+ybias, 20, -5+ybias], [0, 0.3, 0, 0.25, 0, 0, 0.45, 0]]),  # symmetry: [1,0]
           np.array([[-5+xbias, -10+ybias, -5+xbias, -20+ybias], [0, 0, 0.25, 0.3, 0, 0.45, 0, 0]]),  # symmetry: [-1,1]
           np.array([[-5+xbias, 10, -5+xbias, 20], [0.3, 0, 0.25, 0, 0, 0, 0, 0.45]]),  # symmetry: [0,1]
           np.array([[5, -10+ybias, 5, -20+ybias], [0, 0.25, 0, 0.3, 0, 0, 0.45, 0]]),  # symmetry: [1,0]
           np.array([[-10+xbias, -5+ybias, -20+xbias, -5+ybias], [0, 0, 0.3, 0.25, 0, 0.45, 0, 0]])]  # symmetry: [-1,1]
	       ))
	@unpack
	def testGenerateSymmetricData(self, originalDataSet, groundTruth):
		bias = xBoundary[1]
		sheepSpeed = 20
		degrees = [math.pi / 2, 0, math.pi, -math.pi / 2,
		           math.pi / 4, -math.pi * 3 / 4, -math.pi / 4, math.pi * 3 / 4]
		sheepActionSpace = [
			tuple(np.round(sheepSpeed * transitePolarToCartesian(degree))) for
			degree in degrees]
		symmetries = [np.array([1, 1]), np.array([0, 1]), np.array([1, 0]),
		              np.array([-1, 1])]
		generateSymmetricData = GenerateSymmetricData(bias,
		                                              createSymmetricVector,
		                                              symmetries,
		                                              sheepActionSpace)
		sysmetricDataSet = generateSymmetricData(originalDataSet)
		for data in sysmetricDataSet:
			for truthData in groundTruth:
				if np.allclose(data[0], np.array(truthData[0])):
					self.assertSequenceEqual(list(data[1]), list(truthData[1]))


if __name__ == "__main__":
	unittest.main()