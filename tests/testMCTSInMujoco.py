import sys
import os
sys.path.append(os.path.join('..', 'src', 'algorithms'))
sys.path.append(os.path.join('..', 'src'))
import unittest
import numpy as np
from ddt import ddt, data, unpack

# Local import
from mcts import HeuristicDistanceToTarget
from envSheepChaseWolf import GetAgentPos


@ddt
class TestMCTSInMujoco(unittest.TestCase):
    @data((np.asarray([[-8, 0, -8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), -1.6), (np.asarray([[8, 0, 8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), 0), (np.asarray([[10, -10, 10, -10, 0, 0], [-10, 10, -10, 10, 0, 0]]), -2*np.sqrt(2)))
    @unpack
    def testRolloutHeuristicBasedOnClosenessToTarget(self, state, groundTruthReward):
        weight = 0.1
        sheepId = 0
        wolfId = 1
        xPosIndex = 2
        numXPosEachAgent = 2

        getSheepPosition = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
        getWolfPosition = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

        rolloutHeuristic = HeuristicDistanceToTarget(weight, getWolfPosition, getSheepPosition)
        reward = rolloutHeuristic(state)
        self.assertEqual(reward, groundTruthReward)