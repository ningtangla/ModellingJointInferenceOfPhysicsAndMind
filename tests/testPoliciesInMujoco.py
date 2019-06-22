import sys
import os
sys.path.append(os.path.join('..', 'src', 'sheepWolf'))

import unittest
import numpy as np
from ddt import ddt, data, unpack

from policiesFixed import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy, PolicyDirectlyTowardsOtherAgent
from sheepWolfWrapperFunctions import GetAgentPosFromState

@ddt
class TestPoliciesInMujoco(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.xPosIndex = [2, 3]
        self.sheepId = 0
        self.wolfId = 1
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)


    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [4, -3, 4, -3, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), np.asarray((0, 0))))
    @unpack
    def testStationaryAgentPolicy(self, state, groundTruthAction):
        action = stationaryAgentPolicy(state)

        truthValue = np.array_equal(action, groundTruthAction)
        self.assertTrue(truthValue)


    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), 10, np.asarray((-10, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [4, -3, 4, -3, 0, 0]]), 5, np.asarray((-4, 3))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), 1, np.asarray((0, 1))))
    @unpack
    def testPolicyDirectlyTowardsOtherAgent(self, state, actionMagnitude, groundTruthWolfAction):
        policyDirectlyTowardsOtherAgent = PolicyDirectlyTowardsOtherAgent(self.getSheepXPos, self.getWolfXPos,
                                                                          actionMagnitude)
        wolfAction = policyDirectlyTowardsOtherAgent(state)

        truthValue = np.allclose(wolfAction, groundTruthWolfAction)
        self.assertTrue(truthValue)
