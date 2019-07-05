import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
import numpy as np
import exec.compareValueDataStandardizationAndLossCoefs.preprocessTrajectoriesForNN as preprocess


@ddt
class TestGenerateData(unittest.TestCase):
    def setUp(self):
        self.rewardFunc = lambda s, a: 1

    @data((1, [(0, None, None)], [1]),
          (1, [(0, None, None), (1, None, None), (2, None, None)], [3, 2, 1]),
          (0.5, [(0, None, None), (1, None, None), (2, None, None)], [1.75, 1.5, 1]))
    @unpack
    def testAccumulateRewards(self, decay, trajectory, groundTruthRewards):
        accRewards = preprocess.AccumulateRewards(decay, self.rewardFunc)
        rewards = accRewards(trajectory)
        for reward, groundTruthReward in zip(rewards, groundTruthRewards):
            self.assertAlmostEqual(reward, groundTruthReward)

    @data(([(0, 0, {0: 0.9, 1: 0.1})], 1, [(0, 0, {0: 0.9, 1: 0.1}, 1)]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 1,
           [(0, 1, {0: 0.1, 1: 0.9}, 3), (1, 1, {0: 0.6, 1: 0.4}, 2), (2, 0, {0: 0.9, 1: 0.1}, 1)]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 0.5,
           [(0, 1, {0: 0.1, 1: 0.9}, 1.75), (1, 1, {0: 0.6, 1: 0.4}, 1.5), (2, 0, {0: 0.9, 1: 0.1}, 1)]))
    @unpack
    def testAddValuesToTraj(self, traj, decay, groundTruthTrajWithValues):
        trajValueFunc = preprocess.AccumulateRewards(decay, self.rewardFunc)
        trajWithValues = preprocess.addValuesToTraj(traj, trajValueFunc)
        for transition, groundTruthTransition in zip(trajWithValues, groundTruthTrajWithValues):
            self.assertEqual(transition[0:3], groundTruthTransition[0:3])
            self.assertAlmostEqual(transition[3], groundTruthTransition[3])

    @data(([([[0]], 1, {0: 0.5, 1: 0.5}, 1)], [(np.array([0]), 1, {0: 0.5, 1: 0.5}, 1)]),
          ([([[0]], 1, {0: 0.5, 1: 0.5}, 1), ([[1]], 0, {0: 0.5, 1: 0.5}, 1)], [(np.array([0]), 1, {0: 0.5, 1: 0.5}, 1), (np.array([1]), 0, {0: 0.5, 1: 0.5}, 1)]),
          ([([[0, 1], [1, 0]], 1, {0: 0.5, 1: 0.5}, 1)], [(np.array([0, 1, 1, 0]), 1, {0: 0.5, 1: 0.5}, 1)]),
          ([([[0, 1, 2], [1, 0, 3]], 1, {0: 0.5, 1: 0.5}, 1)], [(np.array([0, 1, 2, 1, 0, 3]), 1, {0: 0.5, 1: 0.5}, 1)]))
    @unpack
    def testWorldStatesToNpArrays(self, traj, groundTruthTraj):
        convertedTraj = preprocess.worldStatesToNpArrays(traj)
        for transition, groundTruthTransition in zip(convertedTraj, groundTruthTraj):
            self.assertTrue((transition[0] == groundTruthTransition[0]).all())
            self.assertEqual(transition[1:], groundTruthTransition[1:])

    @data(([([0, 1], 1, {0: 0.5, 1: 0.5}, 1)], [-1, 0, 1], [([0, 1], np.array([0, 0, 1]), {0: 0.5, 1: 0.5}, 1)]),
          ([([0, 1], (-1, 0), {(-1, 0): 0.5, (0, 1): 0.5}, 1)], [(1, 0), (0, 1), (-1, 0), (0, -1)], [([0, 1], np.array([0, 0, 1, 0]), {(-1, 0): 0.5, (0, 1): 0.5}, 1)]))
    @unpack
    def testActionsToLabels(self, traj, actionSpace, groundTruthNewTraj):
        newTraj = preprocess.actionsToLabels(traj, actionSpace)
        for transition, groundTruthTransition in zip(newTraj, groundTruthNewTraj):
            self.assertEqual(transition[0], groundTruthTransition[0])
            self.assertTrue((transition[1] == groundTruthTransition[1]).all())
            self.assertEqual(transition[2:], groundTruthTransition[2:])

    @data(([([0, 1], 1, {0: 0.5, 1: 0.5, -1: 0}, 1)], [-1, 0, 1], [([0, 1], 1, np.array([0, 0.5, 0.5]), 1)]),
          ([([0, 1], (-1, 0), {(-1, 0): 0.5, (0, 1): 0.4, (0, -1): 0.1, (1, 0): 0}, 1)], [(1, 0), (0, 1), (-1, 0), (0, -1)],
           [([0, 1], (-1, 0), np.array([0, 0.4, 0.5, 0.1]), 1)]))
    @unpack
    def testActionDistToProbs(self, traj, actionSpace, groundTruthNewTraj):
        newTraj = preprocess.actionDistsToProbs(traj, actionSpace)
        for transition, groundTruthTransition in zip(newTraj, groundTruthNewTraj):
            self.assertEqual(transition[0:2], groundTruthTransition[0:2])
            self.assertTrue((transition[2] == groundTruthTransition[2]).all())
            self.assertEqual(transition[3], groundTruthTransition[3])


if __name__ == "__main__":
    unittest.main()
