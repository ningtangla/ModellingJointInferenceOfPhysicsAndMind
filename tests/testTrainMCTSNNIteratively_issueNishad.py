import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from ddt import ddt, data, unpack
import unittest
import numpy as np

from exec.trainMCTSNNIteratively.trainMCTSNNIteratively import ProcessTrajectoryForNN, PreProcessTrajectories, QPosInitStdDevForIteration
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory

@ddt
class TestTrainMCTSNNIteratively(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.actionIndex = 1
        self.decay = 1
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all()
                                                         else 0 for index in range(len(self.actionSpace))])
        self.rewardFunction = lambda s, a: 1
        self.accumulateRewards = AccumulateRewards(self.decay, self.rewardFunction)
        self.addValuesToTrajectory = AddValuesToTrajectory(self.accumulateRewards)
        self.getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][1]
        self.removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(self.getTerminalActionFromTrajectory)
        self.processTrajectoryForNN = ProcessTrajectoryForNN(self.actionToOneHot, self.sheepId)
        self.compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                   in zip(tuple1, tuple2)) and len(tuple1) == len(tuple2)
        self.compareTrajectories = lambda traj1, traj2: all(self.compareTuples(tuple1, tuple2) for tuple1, tuple2
                                                            in zip(traj1, traj2)) and len(traj1) == len(traj2)

    # @data(([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], 12)],
    #        [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), 12)]),
    #       ([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], 12),
    #         (np.asarray([[1, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]]), [np.asarray((10, 0)), np.asarray((-7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], 24)],
    #        [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), 12),
    #         (np.asarray([1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), 24)]))
    # @unpack
    # def testProcessTrajectoryForNN(self, trajectory, groundTruthProcessedTrajectory):
    #     processedTrajectory = self.processTrajectoryForNN(trajectory)
    #     truthValue = self.compareTrajectories(processedTrajectory, groundTruthProcessedTrajectory)
    #     self.assertTrue(truthValue)
    #
    #
    # @data(([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], np.asarray([1]))],
    #        [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], np.asarray([1]))]),
    #       ([(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #         [{(0, 10): 1}, {(7, 7): 1}], np.asarray([2])),
    #         (np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), None, None, np.asarray([1]))],
    #        [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}], np.asarray([2]))]))
    # @unpack
    # def testRemoveTerminalTupleFromTrajectory(self, trajectory, groundTruthFilteredTrajectory):
    #     filteredTrajectory = self.removeTerminalTupleFromTrajectory(trajectory)
    #     truthValue = self.compareTrajectories(filteredTrajectory, groundTruthFilteredTrajectory)
    #     self.assertTrue(truthValue)
    #
    # @data(([[(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #          [{(0, 10): 1}, {(7, 7): 1}])],
    #         [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #           [{(0, 10): 1}, {(7, 7): 1}]),
    #          (np.asarray([[1, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]]), [np.asarray((10, 0)), np.asarray((-7, 7))],
    #           [{(0, 10): 1}, {(7, 7): 1}])],
    #         [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))],
    #           [{(0, 10): 1}, {(7, 7): 1}]),
    #          (np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), None, None)]],
    #        [[(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([1]))],
    #         [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([2])),
    #          (np.asarray([1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]), np.asarray([1]))],
    #         [(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0, 0]), np.asarray([2]))]]))
    # @unpack
    # def testPreProcessTrajectories(self, trajectories, groundTruthPreProcessedTrajectories):
    #     preProcessTrajectories = PreProcessTrajectories(self.addValuesToTrajectory,
    #                                                       self.removeTerminalTupleFromTrajectory,
    #                                                       self.processTrajectoryForNN)
    #     preProcessedTrajectories = preProcessTrajectories(trajectories)
    #
    #     truthValue = all(self.compareTrajectories(preProcessedTrajectory, groundTruthPreProcessedTrajectory) for
    #                      preProcessedTrajectory, groundTruthPreProcessedTrajectory in
    #                      zip(preProcessedTrajectories, groundTruthPreProcessedTrajectories))
    #     self.assertTrue(truthValue)
    #
    #
    # @data((0.01, 1), (0.00001, 1000))
    # @unpack
    # def testConstantLearningRateModifier(self, learningRate, trainIteration):
    #     constantLearningRateModifier = ConstantLearningRateModifier(learningRate)
    #     self.assertEqual(constantLearningRateModifier(trainIteration), learningRate)

    @data((200, 5000, 2, 16, 10, 2), (200, 5000, 2, 16, 5000, 16), ((200, 5000, 2, 16, 2000, 58/8)))
    @unpack
    def testQPosInitStdDevForIteration(self, minIterLinearRegion, maxIterLinearRegion, minStdDev, maxStdDev, iteration,
                                       groundTruthStdDev):
        computeQPos = QPosInitStdDevForIteration(minIterLinearRegion, minStdDev, maxIterLinearRegion, maxStdDev)
        stdDev = computeQPos(iteration)
        self.assertAlmostEqual(groundTruthStdDev, stdDev)


if __name__ == '__main__':
    unittest.main()
