import sys
sys.path.append('../exec')
sys.path.append('../src/algorithms')

import unittest
from ddt import ddt, data, unpack
import numpy as np

from mainNeuralNet import GetNonUniformPriorAtSpecificState
from mcts import GetActionPrior


class GetNonUniformPrior:
    def __init__(self, actionSpace, preferredAction, priorForPreferredAction):
        self.actionSpace = actionSpace
        self.preferredAction = preferredAction
        self.priorForPreferredAction = priorForPreferredAction

    def __call__(self, state):
        actionPrior = {action: (1-self.priorForPreferredAction) / (len(self.actionSpace)-1) for action in self.actionSpace}
        actionPrior[self.preferredAction] = self.priorForPreferredAction

        return actionPrior


@ddt
class TestGetNonUniformPriorAtSpecificState(unittest.TestCase):
    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.9),
          (np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (-10, 0), 0.1 / 7),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.125),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (7, 7), 0.125))
    @unpack
    def testGetNonUniformPriorAtSpecificState(self, state, action, groundTruthActionPriorForAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preferredAction = (10, 0)
        priorForPreferredAction = 0.9
        getNonUniformPrior = GetNonUniformPrior(actionSpace, preferredAction, priorForPreferredAction)
        getUniformPrior = GetActionPrior(actionSpace)
        specificState = [[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]
        getNonUniformPriorAtSpecificState = GetNonUniformPriorAtSpecificState(getNonUniformPrior, getUniformPrior, specificState)

        actionPrior = getNonUniformPriorAtSpecificState(state)
        self.assertAlmostEqual(actionPrior[action], groundTruthActionPriorForAction)

