import sys
sys.path.append('../src/sheepWolf')

import unittest
from ddt import ddt, data, unpack

from calculateAngleFunction import *


@ddt
class TestAngleCalculation(unittest.TestCase):
	@data(
		((1,2), (2,4), 0),
		((1,0), (0,1), np.pi/2),
		((1,0), (-1,0), np.pi),
		((2,2), (1,0), np.pi/4)
		)
	@unpack
	def testAngleCalculation(self, firstVector, secondVector, trueAngle):
		calculatedAngle = calculateAngle(firstVector, secondVector)
		self.assertAlmostEqual(calculatedAngle, trueAngle)

	def tearDown(self):
		pass


if __name__ == "__main__":
    angleTest = unittest.TestLoader().loadTestsFromTestCase(TestAngleCalculation)
    unittest.TextTestRunner(verbosity=2).run(angleTest)




