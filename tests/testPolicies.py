import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, HeatSeekingContinuesDeterministicPolicy, ActHeatSeeking, HeatSeekingDiscreteStochasticPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import chooseGreedyAction

@ddt
class TestContinuesActionPolicies(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.xPosIndex = [2, 3]
        self.sheepId = 0
        self.wolfId = 1
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.wolfId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.sheepId, self.posIndex)

    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [4, -3, 4, -3, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), np.asarray((0, 0))))
    @unpack
    def testStationaryAgentPolicy(self, state, groundTruthAction):
        action = chooseGreedyAction(stationaryAgentPolicy(state))

        truthValue = np.array_equal(action, groundTruthAction)
        self.assertTrue(truthValue)

    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), 10, np.asarray((10, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [-4, 3, -4, 3, 0, 0]]), 5, np.asarray((4, -3))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), 1, np.asarray((0, -1))))
    @unpack
    def testHeatSeekingContinuesDeterministicPolicy(self, state, actionMagnitude, groundTruthWolfAction):
        heatSeekingPolicy = HeatSeekingContinuesDeterministicPolicy(self.getSheepXPos, self.getWolfXPos,
                                                                    actionMagnitude)
        action = chooseGreedyAction(heatSeekingPolicy(state))
        truthValue = np.allclose(action, groundTruthWolfAction)
        self.assertTrue(truthValue)

    def tearDown(self):
        pass


@ddt
class TestDiscreteActionPolicies(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.wolfId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.sheepId, self.posIndex)

    @data((np.array([[20, 20], [1, 1]]), np.array((7, 7))),
          (np.array([[20, 20], [80, 80]]), np.array((-7, -7))),
          (np.array([[20, 20], [20, 30]]), np.array((0, -10))))
    @unpack
    def testHeatSeekingDiscreteDeterministicPolicy(self, state, groundTruthAction):
        heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(self.actionSpace, self.getPredatorPos, self.getPreyPos, computeAngleBetweenVectors)
        action = chooseGreedyAction(heatSeekingPolicy(state))
        truthValue = np.allclose(action, groundTruthAction)
        self.assertTrue(truthValue)


@ddt
class TestHeatSeekingDiscreteStochasticPolicy(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.rationalityParam = 0.9
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2

        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)

        self.wolfID = 0
        self.sheepID = 1
        self.masterID = 2
        self.positionIndex = [0, 1]

        self.getWolfPos = GetAgentPosFromState(self.wolfID, self.positionIndex)
        self.getSheepPos = GetAgentPosFromState(self.sheepID, self.positionIndex)
        self.getMasterPos = GetAgentPosFromState(self.masterID, self.positionIndex)

    @data(((3, 2), [(1, 0), (0, 1)], [(-1, 0), (0, -1), (0, 0)]),
          ((0, -1), [(0, -1)], [(-1, 0), (1, 0), (0, 1), (0, 0)]))
    @unpack
    def testHeatSeekingProperAction(self, heatSeekingDirection, trueChosenActions, trueUnchosenActions):
        actionLists = self.actHeatSeeking(heatSeekingDirection)
        chosenActions = actionLists[0]
        unchosenActions = actionLists[1]
        self.assertEqual(chosenActions, trueChosenActions)
        self.assertEqual(unchosenActions, trueUnchosenActions)

    @data(
        ([(2, 3), (4, 2)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(2, 2), (4, 2)], {(-1, 0): 0.1 / 4, (1, 0): 0.9, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(4, 2), (5, 1)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(5, 2), (5, 1)], {(-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.9, (0, 0): 0.1 / 4}),
        ([(4, 2), (2, 3)], {(-1, 0): 0.45, (1, 0): 0.1 / 3, (0, 1): 0.45, (0, -1): 0.1 / 3, (0, 0): 0.1 / 3}),
        ([(4, 2), (2, 2)], {(-1, 0): 0.9, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(6, 6), (6, 10)], {(0, 1): 0.9, (-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4})
    )
    @unpack
    def testHeatSeekingPolicy(self, state, trueActionLikelihood):

        heatSeekingPolicy = HeatSeekingDiscreteStochasticPolicy(self.rationalityParam, self.actHeatSeeking, self.getWolfPos, self.getSheepPos)

        iterationTime = 10000
        trueActionLikelihoodPair = zip(trueActionLikelihood.keys(), trueActionLikelihood.values())
        trueActionCount = {action: trueActionProb * iterationTime for
                           action, trueActionProb in trueActionLikelihoodPair}
        intendedActionList = [heatSeekingPolicy(state) for _ in range(iterationTime)]

        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action], intendedActionList.count(action), delta=200)

    def testRandomPolicy(self):
        state = [[1, 2], [2, 3], [3, 4]]
        randomPolicy = RandomPolicy(self.actionSpace)

        iterationTime = 10000
        trueActionCount = {action: 1 / len(self.actionSpace) * iterationTime for action in self.actionSpace}
        intendedActionList = [randomPolicy(state) for _ in range(iterationTime)]
        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action], intendedActionList.count(action), delta=200)

    def tearDown(self):
        pass


@ddt
class TestHeatSeekingDiscreteStochasticPolicyRandomOrder(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.rationalityParam = 0.9
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2

        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle,
                                             self.upperBoundAngle)
        self.wolfID = 0
        self.sheepID = 2
        self.masterID = 1
        self.positionIndex = [0, 1]

        self.getWolfPos = GetAgentPosFromState(self.wolfID, self.positionIndex)
        self.getSheepPos = GetAgentPosFromState(self.sheepID, self.positionIndex)
        self.getMasterPos = GetAgentPosFromState(self.masterID, self.positionIndex)


    @data(
        ([(2, 3), (1, 1), (4, 2)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(2, 2), (1, 1), (4, 2)], {(-1, 0): 0.1 / 4, (1, 0): 0.9, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(4, 2), (1, 1), (5, 1)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(5, 2), (1, 1), (5, 1)], {(-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.9, (0, 0): 0.1 / 4}),
        ([(4, 2), (1, 1), (2, 3)], {(-1, 0): 0.45, (1, 0): 0.1 / 3, (0, 1): 0.45, (0, -1): 0.1 / 3, (0, 0): 0.1 / 3}),
        ([(4, 2), (1, 1), (2, 2)], {(-1, 0): 0.9, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(6, 6), (1, 1), (6, 10)], {(0, 1): 0.9, (-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4})
    )
    @unpack
    def testHeatSeekingPolicy(self, state, trueActionLikelihood):

        heatSeekingPolicy = HeatSeekingDiscreteStochasticPolicy(self.rationalityParam, self.actHeatSeeking,
                                                                self.getWolfPos, self.getSheepPos)

        iterationTime = 10000
        trueActionLikelihoodPair = zip(trueActionLikelihood.keys(), trueActionLikelihood.values())
        trueActionCount = {action: trueActionProb * iterationTime for
                           action, trueActionProb in trueActionLikelihoodPair}
        intendedActionList = [heatSeekingPolicy(state) for _ in range(iterationTime)]

        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action], intendedActionList.count(action), delta=200)

    def testRandomPolicy(self):
        state = [[1, 2], [2, 3], [3, 4]]
        randomPolicy = RandomPolicy(self.actionSpace)

        iterationTime = 10000
        trueActionCount = {action: 1 / len(self.actionSpace) * iterationTime for action in self.actionSpace}
        intendedActionList = [randomPolicy(state) for _ in range(iterationTime)]

        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action], intendedActionList.count(action), delta=200)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
