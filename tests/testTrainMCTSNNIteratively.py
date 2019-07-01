import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from ddt import ddt, data, unpack
import unittest
import numpy as np

from exec.trainMCTSNNIteratively.trainMCTSNNIteratively import PreProcessTrajectories, ActionToOneHot, \
    AddValuesToTrajectory, AccumulateRewards, PlayToTrain, TrainToPlay, GetPolicy


@ddt
class TestTrainMCTSNNIteratively(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.actionIndex = 1
        self.decay = 1
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.actionToOneHot = ActionToOneHot(self.actionSpace)
        self.rewardFunction = lambda s, a: 1
        self.accumulateRewards = AccumulateRewards(self.decay, self.rewardFunction)
        self.addValuesToTrajectory = AddValuesToTrajectory(self.accumulateRewards)


    @data(([(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), None)],
           [(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], np.asarray([2])),
            (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), None, np.asarray([1]))]),
          ([(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
           (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
            (np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
            (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))])],
           [(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], np.asarray([4])),
            (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], np.asarray([3])),
            (np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], np.asarray([2])),
            (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], np.asarray([1]))]))
    @unpack
    def testAddValuesToTrajectory(self, trajectory, groundTruthTrajectoryWithValues):
        trajectoryWithValues = self.addValuesToTrajectory(trajectory)

        compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                   in zip(tuple1, tuple2))
        compareTrajectories = lambda traj1, traj2: all(compareTuples(tuple1, tuple2) for tuple1, tuple2 in zip(traj1, traj2))

        self.assertTrue(compareTrajectories(trajectoryWithValues, groundTruthTrajectoryWithValues))


    @data(([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)])]]),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)])],
            [(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[1, -1, 1, -1, 0, 0], [2, 2, 2, 2, 0, 0]]), [(7, -7), (0, 10)])]]))
    def testPreProcessTrajectories(self, trajectories):
        preProcessTrajectories = PreProcessTrajectories(self.sheepId, self.actionIndex, self.actionToOneHot,
                                                        self.addValuesToTrajectory)
        trainData = preProcessTrajectories(trajectories)

        states = trainData[0]
        checkStateSize = all(np.size(state) == 12 for state in states)
        self.assertTrue(checkStateSize)

        actions = trainData[1]
        checkActionSize = all(np.size(action) == 8 for action in actions)
        self.assertTrue(checkActionSize)

        checkActionSum = all(np.sum(action) == 1 for action in actions)
        self.assertTrue(checkActionSum)

        checkNumStatesEqualNumActions = len(states) == len(actions)
        self.assertTrue(checkNumStatesEqualNumActions)

        values = trainData[2]
        checkValuesShape = np.shape(values) == (len(states), 1)
        self.assertTrue(checkValuesShape)


    @data((np.asarray((10, 0)), np.asarray((7, 7)), [np.asarray((10, 0)), np.asarray((7, 7))]))
    @unpack
    def testGetPolicy(self, sheepAction, wolfAction, groundTruthAction):
        sheepPolicy = lambda state: sheepAction
        wolfPolicy = lambda state: wolfAction
        getSheepPolicy = lambda NNModel: sheepPolicy
        getWolfPolicy = lambda NNModel: wolfPolicy
        state = np.asarray([[1, 2, 1, 2, 0, 0], [2, 4, 2, 4, 0, 0]])
        getPolicy = GetPolicy(getSheepPolicy, getWolfPolicy)

        policy = getPolicy(None)

        action = policy(state)
        truthValue = np.array_equal(action, groundTruthAction)
        self.assertTrue(truthValue)

if __name__ == '__main__':
    unittest.main()