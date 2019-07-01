import sys
import os
sys.path.append('..')
sys.path.append('../src/neuralNetwork')
import unittest
from ddt import ddt, unpack, data
import numpy as np
from trainTools import learningRateModifier

@ddt
class TestNNparameter(unittest.TestCase):

    @data((0.1, 0.9, 100, 200, 0.081))
    @unpack
    def testLearningRateModifier(self, initLearningRate, decayRate, decayStep, globalStep, groundTruth):
        lrModifier = learningRateModifier(initLearningRate, decayRate, decayStep)
        self.assertAlmostEqual(lrModifier(globalStep), groundTruth, places=5)

if __name__ == "__main__":
    unittest.main()
