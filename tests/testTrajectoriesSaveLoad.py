import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd

from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf
from exec.evaluationFunctions import conditionDfFromParametersDict


@ddt
class TestTrajectoriesSaveLoad(unittest.TestCase):
    def setUp(self):
        self.stateIndex = 0
        self.getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][self.stateIndex])[0]))
        self.getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
        self.getAllLevelValuesRange = {'timeStep': self.getRangeTrajectoryLength, 'agentId': self.getRangeNumAgentsFromTrajectory}
        self.getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 2)
        self.getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 3)
        self.getAgentVelXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 4)
        self.getAgentVelYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 5)
        self.extractColumnValues = {'xPos': self.getAgentPosXCoord, 'yPos': self.getAgentPosYCoord,
                                    'xVel': self.getAgentVelXCoord, 'yVel': self.getAgentVelYCoord}
        self.convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(self.getAllLevelValuesRange,
                                                                     conditionDfFromParametersDict,
                                                                     self.extractColumnValues)
    @data(([(np.asarray([[1, 2, 1, 2, 3, 4], [5, 6, 5, 6, 7, 8]]), [np.asarray((0, 10)), np.asarray((7, 7))]),
            (np.asarray([[-1, -2, -1, -2, -3, -4], [-5, -6, -5, -6, -7, -8]]), [np.asarray((0, 10)), np.asarray((7, 7))])],
           pd.DataFrame([(1, 2, 3, 4), (5, 6, 7, 8), (-1, -2, -3, -4),(-5, -6, -7, -8)],
                        index = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=['timeStep', 'agentId']),
                        columns = ('xPos', 'yPos', 'xVel', 'yVel'))))
    @unpack
    def testConvertTrajectoryToStateDf(self, trajectory, groundTruthDf):
        df = self.convertTrajectoryToStateDf(trajectory)
        print(df)
        truthValue = groundTruthDf.equals(df)
        self.assertTrue(truthValue)



if __name__ == '__main__':
    unittest.main()