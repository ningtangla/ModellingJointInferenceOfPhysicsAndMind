import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from ddt import ddt, data, unpack
import unittest
import numpy as np

from exec.preProcessing import AccumulateRewards, AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories


@ddt
class TestPreProcessing(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.actionIndex = 1
        self.decay = 1
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all()
                                                         else 0 for index in range(len(self.actionSpace))])
        self.rewardFunction = lambda s, a: 1
        self.anotherRewardFunction = lambda s, a: -1
        self.accumulateRewards = AccumulateRewards(self.decay, self.rewardFunction) 
        self.accumulateMultipleAgentRewards = AccumulateMultiAgentRewards(self.decay, [self.rewardFunction, self.anotherRewardFunction]) 
        self.addValuesToTrajectory = AddValuesToTrajectory(self.accumulateRewards)
        self.getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][1]
        self.removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(self.getTerminalActionFromTrajectory)
        self.processTrajectoryForPolicyValueNet = ProcessTrajectoryForPolicyValueNet(self.actionToOneHot, self.sheepId)
        self.compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                        in zip(tuple1, tuple2)) and len(tuple1) == len(tuple2)
        self.compareTrajectories = lambda traj1, traj2: all(self.compareTuples(tuple1, tuple2) for tuple1, tuple2
                                                            in zip(traj1, traj2)) and len(traj1) == len(traj2)

    @data(([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.array([12]))],
           [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.array([12]))]),
          ([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.array([12])),
            (np.asarray([[1, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]]), [np.asarray((10, 0)), np.asarray((-7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.array([24]))],
           [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.array([12])),
            (np.asarray([1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), np.array([24]))]))
    @unpack
    def testProcessTrajectoryForPolicyValueNet(self, trajectory, groundTruthProcessedTrajectory):
        processedTrajectory = self.processTrajectoryForPolicyValueNet(trajectory)
        truthValue = self.compareTrajectories(processedTrajectory, groundTruthProcessedTrajectory)
        self.assertTrue(truthValue)

    @data(([[(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
              [{(0, 10): 1}, {(7, 7): 1}])],
            [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
              [{(0, 10): 1}, {(7, 7): 1}]),
             (np.asarray([[1, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]]), [np.asarray((10, 0)), np.asarray((-7, 7))],
              [{(0, 10): 1}, {(7, 7): 1}])],
            [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
              [{(0, 10): 1}, {(7, 7): 1}]),
             (np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), None, None)]],
           [[(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([1]))],
            [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([2])),
             (np.asarray([1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), np.asarray([1]))],
            [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([2]))]]))
    @unpack
    def testPreProcessTrajectories(self, trajectories, groundTruthPreProcessedTrajectories):
        preProcessTrajectories = PreProcessTrajectories(self.addValuesToTrajectory,
                                                        self.removeTerminalTupleFromTrajectory,
                                                        self.processTrajectoryForPolicyValueNet)
        preProcessedTrajectories = preProcessTrajectories(trajectories)
        truthValue = all(self.compareTrajectories(preProcessedTrajectory, groundTruthPreProcessedTrajectory) for
                         preProcessedTrajectory, groundTruthPreProcessedTrajectory in
                         zip(preProcessedTrajectories, groundTruthPreProcessedTrajectories))
        self.assertTrue(truthValue)

    @data(([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.asarray([1]))],
           [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.asarray([1]))]),
          ([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.asarray([2])),
            (np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), None, None, np.asarray([1]))],
           [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
             [{(0, 10): 1}, {(7, 7): 1}], np.asarray([2]))]))
    @unpack
    def testRemoveTerminalTupleFromTrajectory(self, trajectory, groundTruthFilteredTrajectory):
        filteredTrajectory = self.removeTerminalTupleFromTrajectory(trajectory)
        truthValue = self.compareTrajectories(filteredTrajectory, groundTruthFilteredTrajectory)
        self.assertTrue(truthValue)

    @data((1, [(0, None, None)], [1]),
          (1, [(0, None, None), (1, None, None), (2, None, None)], [3, 2, 1]),
          (0.5, [(0, None, None), (1, None, None), (2, None, None)], [1.75, 1.5, 1]))
    @unpack
    def testAccumulateRewards(self, decay, trajectory, groundTruthRewards):
        accRewards = AccumulateRewards(decay, self.rewardFunction)
        rewards = accRewards(trajectory)
        for reward, groundTruthReward in zip(rewards, groundTruthRewards):
            self.assertAlmostEqual(reward, groundTruthReward)
    
    @data((1, [(0, None, None)], [(1, -1)]),
          (1, [(0, None, None), (1, None, None), (2, None, None)], [(3,-3), (2, -2), (1,-1)]),
          (0.5, [(0, None, None), (1, None, None), (2, None, None)], [(1.75, -1.75), (1.5, -1.5), (1, -1)]))
    @unpack
    def testAccumulateMultiAgentRewards(self, decay, trajectory, groundTruthRewards):
        accRewards = AccumulateMultiAgentRewards(decay, [self.rewardFunction, self.anotherRewardFunction])
        rewards = accRewards(trajectory)
        for reward, groundTruthReward in zip(rewards, groundTruthRewards):
            self.assertTrue(np.all(np.array(reward) == np.array(groundTruthReward)))

    @data(([(0, 0, {0: 0.9, 1: 0.1})], 1, [(0, 0, {0: 0.9, 1: 0.1}, np.array([1]))]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 1,
           [(0, 1, {0: 0.1, 1: 0.9}, 3), (1, 1, {0: 0.6, 1: 0.4}, 2), (2, 0, {0: 0.9, 1: 0.1}, np.array([1]))]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 0.5,
           [(0, 1, {0: 0.1, 1: 0.9}, 1.75), (1, 1, {0: 0.6, 1: 0.4}, 1.5), (2, 0, {0: 0.9, 1: 0.1}, np.array([1]))]))
    @unpack
    def testAddValuesToTraj(self, traj, decay, groundTruthTrajWithValues):
        self.accumulateRewards = AccumulateRewards(decay, self.rewardFunction)
        self.addValuesToTrajectory = AddValuesToTrajectory(self.accumulateRewards)
        trajWithValues = self.addValuesToTrajectory(traj)
        for transition, groundTruthTransition in zip(trajWithValues, groundTruthTrajWithValues):
            self.assertEqual(transition[0:4], groundTruthTransition[0:4])
            #self.assertAlmostEqual(transition[3], groundTruthTransition[3])

    @data(([(0, 0, {0: 0.9, 1: 0.1})], 1, [(0, 0, {0: 0.9, 1: 0.1}, np.array([1,-1]))]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 1,
           [(0, 1, {0: 0.1, 1: 0.9}, np.array([3,-3])), (1, 1, {0: 0.6, 1: 0.4}, np.array([2, -2])), (2, 0, {0: 0.9, 1: 0.1}, np.array([1, -1]))]),
          ([(0, 1, {0: 0.1, 1: 0.9}), (1, 1, {0: 0.6, 1: 0.4}), (2, 0, {0: 0.9, 1: 0.1})], 0.5,
           [(0, 1, {0: 0.1, 1: 0.9}, np.array([1.75, -1.75])), (1, 1, {0: 0.6, 1: 0.4}, np.array([1.5, -1.5])), (2, 0, {0: 0.9, 1: 0.1}, np.array([1, -1]))]))
    @unpack
    def testAddMultiAgentValuesToTraj(self, traj, decay, groundTruthTrajWithValues):
        accRewards = AccumulateMultiAgentRewards(decay, [self.rewardFunction, self.anotherRewardFunction])
        self.addValuesToTrajectory = AddValuesToTrajectory(accRewards)
        trajWithValues = self.addValuesToTrajectory(traj)
        for transition, groundTruthTransition in zip(trajWithValues, groundTruthTrajWithValues):
            self.assertEqual(transition[0:3], groundTruthTransition[0:3])
            self.assertTrue(np.all(transition[3] == groundTruthTransition[3]))


if __name__ == '__main__':
    unittest.main()
