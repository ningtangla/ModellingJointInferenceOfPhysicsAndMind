import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


@ddt
class TestWrapperFunctions(unittest.TestCase):
    @data((0, [2, 3], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]]), np.asarray([1, 2])),
          (1, [2, 3], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]]), np.asarray([3, 4])))
    @unpack
    def testGetAgentPosFromState(self, agentId, posIndex, state, groundTruthAgentPos):
        getAgentPosFromState = GetAgentPosFromState(agentId, posIndex)
        agentPos = getAgentPosFromState(state)

        truthValue = np.array_equal(agentPos, groundTruthAgentPos)
        self.assertTrue(truthValue)

    @unittest.skip
    @data((0, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                 {(0, 0): 0}]),
               (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                [{(10, 0): 0, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 1, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                 {(0, 0): 0}])], np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]])),
          (1, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                 {(0, 0): 0}]),
               (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                [{(10, 0): 0, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 1, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                 {(0, 0): 0}])], np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]])))
    @unpack
    def testGetStateFromTrajectory(self, timeStep, trajectory, groundTruthState):
        stateIndex = 0
        getStateFromTrajectory = GetStateFromTrajectory(timeStep, stateIndex)
        state = getStateFromTrajectory(trajectory)

        truthValue = np.array_equal(state, groundTruthState)
        self.assertTrue(truthValue)


    @unittest.skip
    @data((0, 0, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                   [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 1}]),
                  (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                   [{(10, 0): 0, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 1, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 0}])], np.asarray([3, 4])),
          (1, 1, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                   [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 1}]),
                  (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                   [{(10, 0): 0, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 1, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 0}])], np.asarray([-6, 8])))
    @unpack
    def testGetAgentPosFromTrajectory(self, agentId, timeStep, trajectory, groundTruthAgentPos):
        xPosIndex = [2, 3]
        stateIndex = 0
        getAgentPosFromState = GetAgentPosFromState(agentId, xPosIndex)
        getStateFromTrajectory = GetStateFromTrajectory(timeStep, stateIndex)
        getAgentPosFromTrajectory = GetAgentPosFromTrajectory(getAgentPosFromState, getStateFromTrajectory)
        agentPos = getAgentPosFromTrajectory(trajectory)

        truthValue = np.array_equal(agentPos, groundTruthAgentPos)
        self.assertTrue(truthValue)


    @unittest.skip
    @data((1, 1, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                   [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 1}]),
                  (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                  [{(10, 0): 0, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 1, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                   {(0, 0): 1}])], np.asarray((0, 0))),
          (0, 0, [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))],
                   [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 1}]),
                  (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))],
                   [{(10, 0): 1, (0, 10): 0, (-10, 0): 0, (0, -10): 0, (7, 7): 0, (7, -7): 0, (-7, 7): 0, (-7, -7): 0},
                    {(0, 0): 1}])], np.asarray((10, 0))))
    @unpack
    def testGetAgentActionFromTrajectory(self, timeStep, agentId, trajectory, groundTruthAgentAction):
        actionIndex = 1
        getAgentActionFromTrajectory = GetAgentActionFromTrajectory(timeStep, actionIndex, agentId)
        agentAction = getAgentActionFromTrajectory(trajectory)

        truthValue = np.array_equal(agentAction, groundTruthAgentAction)
        self.assertTrue(truthValue)


if __name__ == "__main__":
    unittest.main()
