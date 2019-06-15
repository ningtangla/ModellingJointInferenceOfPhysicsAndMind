import unittest
from ddt import ddt, data, unpack
import numpy as np
from dataTools import generateSymmetricData

@ddt
class TestGenerateData(unittest.TestCase):
	# (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)
	@data(([np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0], [20]])],
		   [np.array([[10, 5, 20, 5], [0.25, 0.3, 0, 0, 0.45, 0, 0, 0], [20]]),
		   np.array([[5, 10, 5, 20], [0.3, 0.25, 0, 0, 0.45, 0, 0, 0], [20]]), # symmetry: [1,1]
           np.array([[-10, 5, -20, 5], [0.25, 0, 0.3, 0, 0, 0, 0, 0.45], [20]]),  # symmetry: [0,1]
           np.array([[10, -5, 20, -5], [0, 0.3, 0, 0.25, 0, 0, 0.45, 0], [20]]),  # symmetry: [1,0]
           np.array([[-5, -10, -5, -20], [0, 0, 0.25, 0.3, 0, 0.45, 0, 0], [20]]),  # symmetry: [-1,1]
	       # use data get with symmetry [1,1]
           np.array([[-5, 10, -5, 20], [0.3, 0, 0.25, 0, 0, 0, 0, 0.45], [20]]),  # symmetry: [0,1]
           np.array([[5, -10, 5, -20], [0, 0.25, 0, 0.3, 0, 0, 0.45, 0], [20]]),  # symmetry: [1,0]
           np.array([[-10, -5, -20, -5], [0, 0, 0.3, 0.25, 0, 0.45, 0, 0], [20]])]  # symmetry: [-1,1]
	       ))
	@unpack
	def testGenerateSymmetricData(self, originalDataSet, groundTruth):
		sysmetricDataSet = generateSymmetricData(originalDataSet)
		for data in sysmetricDataSet:
			for truthData in groundTruth:
				if (np.array(data[0]) == np.array(truthData[0])).all():
					self.assertEqual(list(data[1]), list(truthData[1]))

if __name__ == "__main__":
	unittest.main()


