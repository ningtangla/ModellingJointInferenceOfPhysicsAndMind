import sys
import os
sys.path.append('..')
sys.path.append(os.path.join('..', 'src'))
sys.path.append(os.path.join('..', 'src', 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join('..', 'src', 'algorithms'))
sys.path.append(os.path.join('..', 'src', 'neuralNetwork'))
sys.path.append(os.path.join('..', 'exec', 'testMCTSUniformVsNNPriorChaseMujoco'))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from exec.testMCTSUniformVsNNPriorChaseMujoco.testMCTSUniformVsNNPriorChaseMujoco import GetNonUniformPriorAtSpecificState, GenerateTrajectories
from src.algorithms.mcts import GetActionPrior
from exec.testMCTSUniformVsNNPriorChaseMujoco.trainNeuralNet import ActionToOneHot, PreProcessTrajectories


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

        getUniformPrior = GetActionPrior(actionSpace)
        specificState = [[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]
        getNonUniformPriorAtSpecificState = GetNonUniformPriorAtSpecificState(getNonUniformPrior, getUniformPrior, specificState)

        actionPrior = getNonUniformPriorAtSpecificState(state)
        self.assertAlmostEqual(actionPrior[action], groundTruthActionPriorForAction)


    @data(((10, 0), [1, 0, 0, 0, 0, 0, 0, 0]), ((7, 7), [0, 1, 0, 0, 0, 0, 0, 0]), ((1, 2), [0, 0, 0, 0, 0, 0, 0, 0]),
          ((-7, -7), [0, 0, 0, 0, 0, 1, 0, 0]))
    @unpack
    def testActionToOneHot(self, action, groundTruthOneHotAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        oneHotAction = actionToOneHot(action)

        self.assertEqual(oneHotAction, groundTruthOneHotAction)


    @data(([[([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)]), ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], None)]],
          [([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])]),
          ([[([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)]),
             ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)])]],
           [([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
            ([-3, 0, -3, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])]))
    @unpack
    def testPreProcessTrajectories(self, trajectories, groundTruthStateActionPairsProcessed):
        sheepId = 0
        actionIndex = 1
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
        stateActionPairsProcessed = preProcessTrajectories(trajectories)

        self.assertEqual(stateActionPairsProcessed, groundTruthStateActionPairsProcessed)

if __name__ == "__main__":
    unittest.main()
