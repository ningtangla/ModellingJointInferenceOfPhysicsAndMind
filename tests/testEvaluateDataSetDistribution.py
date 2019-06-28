import sys
sys.path.append("../exec")
import unittest
from ddt import ddt, data, unpack
import numpy as np
from evaluateDataSetDistribution import GetAgentStateFromDataSetState


@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
    @data((0, 2, [1,2,3,4], [1,2]), (1, 2, [1,2,3,4], [3, 4]))
    @unpack
    def testGetAgentStateFromDataSetState(self, agentID, agentStateDim, dataSetState, groundTruth):
        getAgentStateFromDataSetState = GetAgentStateFromDataSetState(agentID, agentStateDim)
        self.assertTrue((np.array(getAgentStateFromDataSetState(dataSetState)) == np.array(groundTruth)).all())


if __name__ == "__main__":
    unittest.main()