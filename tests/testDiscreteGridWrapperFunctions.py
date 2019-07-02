import sys
sys.path.append('../src/sheepWolf')

import unittest
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromTrajectory


@ddt
class TestWrapperFunc(unittest.TestCase):
    @data((1, [0,1], [[1,2],[3,4]], [3,4]),
          (0, [0,1], [[1,2],[3,4]], [1,2]))
    @unpack
    def testAgentPositionRetrieval(self, agentID, positionIndex, state, trueAgentPosition):
        getAgentPosition = GetAgentPosFromTrajectory(agentID, positionIndex)
        agentPosition = getAgentPosition(state)
        self.assertEqual(agentPosition, trueAgentPosition)

    def tearDown(self):
        pass


if __name__ == "__main__":
    wrapperFunctionTest = unittest.TestLoader().loadTestsFromTestCase(TestWrapperFunc)
    unittest.TextTestRunner(verbosity=2).run(wrapperFunctionTest)
