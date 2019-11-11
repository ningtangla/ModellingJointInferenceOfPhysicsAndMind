import sys
sys.path.append("../exec")
import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd
from evaluateDataSetDistribution import GetAgentStateFromDataSetState
from evaluateDataSetDistribution import isin

@ddt
class TestAnalyticGeometryFunctions(unittest.TestCase):
    def setUp(self):
        xSections = [(2, 5)]
        ySections = [(2, 5)]
        self.levelNames = ["xSection", "ySection"]
        self.levelValues = [xSections, ySections]
        MultiIndex = pd.MultiIndex.from_product(self.levelValues, names=self.levelNames)
        self.df = pd.DataFrame(index=MultiIndex)

    @data((0, 2, [1,2,3,4], [1,2]), (1, 2, [1,2,3,4], [3, 4]))
    @unpack
    def testGetAgentStateFromDataSetState(self, agentID, agentStateDim, dataSetState, groundTruth):
        getAgentStateFromDataSetState = GetAgentStateFromDataSetState(agentStateDim)
        self.assertTrue((np.array(getAgentStateFromDataSetState(dataSetState, agentID)) == np.array(groundTruth)).all())

    @data(([0, 10], 5, True), ([0, 10], 11, False))
    @unpack
    def testIsin(self, range, number, groundTruth):
        self.assertEqual(isin(range, number), groundTruth)


if __name__ == "__main__":
    unittest.main()
