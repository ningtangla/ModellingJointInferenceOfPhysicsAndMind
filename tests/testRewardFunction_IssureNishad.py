import sys
import os
sys.path.append('..')

import unittest
from ddt import ddt, unpack, data
import numpy as np

from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget


@ddt
class TestMeasurementFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.killzoneRadius = 0.5

    @data((np.asarray([[-8, 0, -8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), -1.6), (np.asarray([[8, 0, 8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), 0), (np.asarray([[10, -10, 10, -10, 0, 0], [-10, 10, -10, 10, 0, 0]]), -2 * np.sqrt(2)))
    @unpack
    def testRolloutHeuristicBasedOnClosenessToTarget(self, state, groundTruthReward):
        weight = 0.1

        rolloutHeuristic = HeuristicDistanceToTarget(
            weight, self.getWolfXPos, self.getSheepXPos)
        reward = rolloutHeuristic(state)
        self.assertEqual(reward, groundTruthReward)


if __name__ == "__main__":
    unittest.main()
