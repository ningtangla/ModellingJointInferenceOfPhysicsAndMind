import sys
import os
sys.path.append('..')

import unittest
import numpy as np
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, HeatSeekingContinuesDeterministicPolicy, ActHeatSeeking, HeatSeekingDiscreteStochasticPolicy
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
@ddt
class TestContinuesStatePolicies(unittest.TestCase):
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


    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), 10, np.asarray((10, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [-4, 3, -4, 3, 0, 0]]), 5, np.asarray((4, -3))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), 1, np.asarray((0, -1))))
    @unpack
    def testHeatSeekingContinuesDeterministicPolicy(self, state, actionMagnitude, groundTruthWolfAction):
        heatSeekingPolicy = HeatSeekingContinuesDeterministicPolicy(self.getSheepXPos, self.getWolfXPos,
                                                                          actionMagnitude)
        action = heatSeekingPolicy(state)
        truthValue = np.allclose(action, groundTruthWolfAction)
        self.assertTrue(truthValue)

    def tearDown(self):
        pass

@ddt
class TestPolicyFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]
        self.rationalityParam = 0.9
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi/2

        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)

        self.wolfID = 0
        self.sheepID = 1
        self.masterID = 2
        self.positionIndex = [0, 1]

        self.locateWolf = GetAgentPosFromState(self.wolfID, self.positionIndex)
        self.locateSheep = GetAgentPosFromState(self.sheepID, self.positionIndex)
        self.locateMaster = GetAgentPosFromState(self.masterID, self.positionIndex)

    @data(((3,2),[(1,0), (0,1)],[(-1, 0), (0, -1), (0, 0)]),
           ((0,-1), [(0, -1)],[(-1, 0), (1, 0), (0, 1), (0, 0)]))
    @unpack 
    def testHeatSeekingProperAction(self, heatSeekingDirection, trueChosenActions, trueUnchosenActions):
        actionLists = self.actHeatSeeking(heatSeekingDirection)
        chosenActions = actionLists[0]
        unchosenActions = actionLists[1]
        self.assertEqual(chosenActions, trueChosenActions)
        self.assertEqual(unchosenActions, trueUnchosenActions)

    @data(
        ([(2, 3),(4, 2)], {(-1, 0): 0.1/3, (1,0): 0.45, (0, 1): 0.1/3, (0, -1): 0.45, (0,0): 0.1/3}),
        ([(2, 2), (4, 2)], {(-1, 0): 0.1 / 4, (1, 0): 0.9, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(4, 2),(5, 1)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(5, 2), (5, 1)], {(-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.9, (0, 0): 0.1 / 4}),
        ([(4, 2), (2, 3)], {(-1, 0): 0.45, (1, 0): 0.1 / 3, (0, 1): 0.45, (0, -1): 0.1 / 3, (0, 0): 0.1 / 3}),
        ([(4, 2), (2, 2)], {(-1, 0): 0.9, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}))
    @unpack
    def testHeatSeekingPolicy(self, state, trueActionLikelihood):

        heatSeekingPolicy = HeatSeekingDiscreteStochasticPolicy(self.rationalityParam, self.actHeatSeeking, self.locateWolf, self.locateSheep)

        iterationTime = 10000
        trueActionLikelihoodPair = zip(trueActionLikelihood.keys(), trueActionLikelihood.values())
        trueActionCount = {action: trueActionProb * iterationTime for
                                  action, trueActionProb in trueActionLikelihoodPair}
        intendedActionList = [heatSeekingPolicy(state) for _ in range(iterationTime)]

        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action],intendedActionList.count(action), delta=200)
    
    def testRandomPolicy(self):
        state = [[1,2], [2,3], [3,4]]
        randomPolicy = RandomPolicy(self.actionSpace)

        iterationTime = 10000
        trueActionCount = {action: 1/len(self.actionSpace) * iterationTime for action in self.actionSpace}
        intendedActionList = [randomPolicy(state) for _ in range(iterationTime)] 
        
        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action],intendedActionList.count(action), delta=200)



    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
