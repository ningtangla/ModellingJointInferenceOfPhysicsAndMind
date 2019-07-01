import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import numpy as np

import unittest
from ddt import ddt, data, unpack

from exec.testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco.trainNeuralNet import PreProcessTrajectories, ActionToOneHot, \
    AccumulateRewards


@ddt
class TestEvaluateMCTSPolicyValue(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.actionIndex = 1
        self.rewardFunction = lambda state, action: 1


    @data(((10, 0), np.asarray([1, 0, 0, 0, 0, 0, 0, 0])), ((-7, 7), np.asarray([0, 0, 0, 1, 0, 0, 0, 0])),
          ((0, -10), np.asarray([0, 0, 0, 0, 0, 0, 1, 0])))
    @unpack
    def testActionToOneHot(self, action, groundTruthOneHot):
        actionToOneHot = ActionToOneHot(self.actionSpace)
        oneHot = actionToOneHot(action)
        truthValue = np.array_equal(oneHot, groundTruthOneHot)

        self.assertTrue(truthValue)


    @data((1, [(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], None)], np.asarray([2, 1])),
          (0.99, [(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)])], np.asarray([1.99, 1])))
    @unpack
    def testAccumulateRewards(self, decay, trajectory, groundTruthRewards):
        accumulateRewards = AccumulateRewards(decay, self.rewardFunction)
        rewards = accumulateRewards(trajectory)
        truthValue = np.array_equal(groundTruthRewards, rewards)

        self.assertTrue(truthValue)


    @data(([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)], 2),
             ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], None, 1)]],
          [(np.asarray([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0])), 2]),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)], 32),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)], 56)]],
           [(np.asarray([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), 32),
            (np.asarray([-3, 0, -3, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), 56)]))
    @unpack
    def testPreProcessTrajectories(self, trajectories, groundTruthTriples):
        actionToOneHot = ActionToOneHot(self.actionSpace)
        preProcessTrajectories = PreProcessTrajectories(self.sheepId, self.actionIndex, actionToOneHot)

        processedTriples = preProcessTrajectories(trajectories)
        compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                   in zip(tuple1, tuple2))
        processedAndGroundTruthTriples = zip(processedTriples, groundTruthTriples)
        truthValue = all(compareTuples(processedTriple, groundTruthTriple) for processedTriple, groundTruthTriple in
                         processedAndGroundTruthTriples)

        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()

