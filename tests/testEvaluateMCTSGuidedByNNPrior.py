import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append('..')

import unittest
from ddt import ddt, data, unpack
import numpy as np

from exec.testMCTSUniformVsNNPriorSheepChaseWolfMujoco.testMCTSUniformVsNNPriorSheepChaseWolfMujoco import \
    GetNonUniformPriorAtSpecificState
from exec.testMCTSUniformVsNNPriorSheepChaseWolfMujoco.trainNeuralNet import ActionToOneHot, PreProcessTrajectories


@ddt
class TestNNMCTS(unittest.TestCase):
    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.9),
          (np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (-10, 0), 0.1 / 7),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.125),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (7, 7), 0.125))
    @unpack
    def testGetNonUniformPriorAtSpecificState(self, state, action, groundTruthActionPriorForAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preferredAction = (10, 0)
        priorForPreferredAction = 0.9

        actionPriorNonUniform = {action: (1-priorForPreferredAction) / (len(actionSpace)-1)
                                 for action in actionSpace}
        actionPriorNonUniform[preferredAction] = priorForPreferredAction
        getNonUniformPrior = lambda state: actionPriorNonUniform

        uniformPrior = {action: 1/len(actionSpace) for action in actionSpace}
        getUniformPrior = lambda state: uniformPrior
        specificState = [[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]
        getNonUniformPriorAtSpecificState = GetNonUniformPriorAtSpecificState(getNonUniformPrior, getUniformPrior, specificState)
        actionPrior = getNonUniformPriorAtSpecificState(state)

        self.assertAlmostEqual(actionPrior[action], groundTruthActionPriorForAction)


    @data(((10, 0), np.asarray([1, 0, 0, 0, 0, 0, 0, 0])), ((7, 7), np.asarray([0, 1, 0, 0, 0, 0, 0, 0])),
          ((1, 2), np.asarray([0, 0, 0, 0, 0, 0, 0, 0])), ((-7, -7), np.asarray([0, 0, 0, 0, 0, 1, 0, 0])))
    @unpack
    def testActionToOneHot(self, action, groundTruthOneHotAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        oneHotAction = actionToOneHot(action)
        truthValue = np.array_equal(oneHotAction, groundTruthOneHotAction)

        self.assertTrue(truthValue)


    @data(([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), None)]],
          [(np.asarray([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]))]),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)]),
             (np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [(10, 0), (0, 0)])]],
           [(np.asarray([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0])),
            (np.asarray([-3, 0, -3, 0, 0, 0, 4, 0, 4, 0, 0, 0]), np.asarray([1, 0, 0, 0, 0, 0, 0, 0]))]))
    @unpack
    def testPreProcessTrajectories(self, trajectories, groundTruthStateActionPairsProcessed):
        sheepId = 0
        actionIndex = 1
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
        stateActionPairsProcessed = preProcessTrajectories(trajectories)

        compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                   in zip(tuple1, tuple2))
        processedAndGroundTruthPairs = zip(stateActionPairsProcessed, groundTruthStateActionPairsProcessed)
        truthValue = all(compareTuples(processedPair, groundTruthPair) for processedPair, groundTruthPair in
                         processedAndGroundTruthPairs)

        self.assertTrue(truthValue)


if __name__ == "__main__":
    unittest.main()
