import sys
import os
sys.path.append('..')
sys.path.append('../src/neuralNetwork')
import unittest
from ddt import ddt, unpack, data
import numpy as np
from trainTools import LearningRateModifier, SampleBatchFromTrajectory

@ddt
class TestNNparameter(unittest.TestCase):
    def setUp(self):
        self.sheepID = 0
        self.preference = False
        self.trajStateIndex = 0
        self.trajActionDistIndex = 2
        self.trajValueIndex = 3

    @data((0.1, 0.9, 100, 200, 0.081))
    @unpack
    def testLearningRateModifier(self, initLearningRate, decayRate, decayStep, globalStep, groundTruth):
        lrModifier = LearningRateModifier(initLearningRate, decayRate, decayStep)
        self.assertAlmostEqual(lrModifier(globalStep), groundTruth, places=5)

    @data((1, 1, [[[[[1, 2], [1, 2]],
                    [[1, 1], [1, 1]],
                    [{'a':0.1, 'b':0.1, 'c':0.1, 'd':0.1, 'f':0.1, 'g':0.1, 'h':0.1, 'i':0.3}, (1,1)],
                    [0.3]],
                   [[[[1, 2], [1, 2]],
                    [[1, 1], [1, 1]],
                    [{'a':0.1, 'b':0.1, 'c':0.1, 'd':0.1, 'f':0.1, 'g':0.1, 'h':0.1, 'i':0.3}, (1,1)],
                    [0.3]]]]], [[1,2, 1,2]]))
    @unpack
    def testSampleBatchFromTrajectory(self, trajNum, stepNum, dataSet, groundTruth):
        sample = SampleBatchFromTrajectory(self.sheepID, self.preference,
                                           self.trajStateIndex, self.trajActionDistIndex,
                                           self.trajValueIndex, trajNum, stepNum)
        state, actionDist, value = sample(dataSet)
        self.assertTrue((np.array(state) == np.array(groundTruth)).all())

if __name__ == "__main__":
    unittest.main()
