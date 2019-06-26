import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os
sys.path.append('..')

# Local import
from src.constrainedChasingEscapingEnv.envNoPhysics import Reset, TransiteForNoPhysics, IsTerminal, StayInBoundaryByReflectVelocity, CheckBoundary
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeVectorNorm

@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        self.numOfAgent = 2
        self.sheepId = 0
        self.wolfId = 1
        self.posIndex = [0, 1]
        self.xBoundary = [0, 640]
        self.yBoundary = [0, 480]
        self.minDistance = 50
        self.getSheepPos = GetAgentPosFromState(
            self.sheepId, self.posIndex)
        self.getWolfPos = GetAgentPosFromState(
            self.wolfId, self.posIndex)
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)
        self.isTerminal = IsTerminal(
            self.getSheepPos, self.getWolfPos, self.minDistance, computeVectorNorm)
        self.transition = TransiteForNoPhysics(self.stayInBoundaryByReflectVelocity)

    @data((np.array([[0, 0], [0, 0]]), [0, 0], np.array([[0, 0], [0, 0]])), (np.array([[9, 5], [2, 7]]), [0, 0], np.array([[9, 5], [2, 7]])))
    @unpack
    def testReset(self, initPosition, initPositionNoise, groundTruthReturnedInitialState):
        reset = Reset(self.numOfAgent, initPosition, initPositionNoise)
        returnedInitialState = reset()
        truthValue = returnedInitialState == groundTruthReturnedInitialState
        self.assertTrue(truthValue.all())

    @data((np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])), (np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[2, 2], [3, 5]])))
    @unpack
    def testTransition(self, state, action, groundTruthReturnedNextState):
        nextState = self.transition(state, action)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())

    @data(([[2, 2], [10, 10]], True), ([[10, 23], [100, 100]], False))
    @unpack
    def testIsTerminal(self, state, groundTruthTerminal):
        terminal = self.isTerminal(state)
        self.assertEqual(terminal, groundTruthTerminal)

    @data(([0, 0], [0, 0], [0, 0]), ([1, -2], [1, -3], [1, 2]), ([1, 3], [2, 2], [1, 3]))
    @unpack
    def testCheckBoundaryAndAdjust(self, state, action, groundTruthNextState):
        checkState, checkAction = self.stayInBoundaryByReflectVelocity(state, action)
        truthValue = checkState == groundTruthNextState
        self.assertTrue(truthValue.all())

    @data(([1, 1], True), ([1, -2], False), ([650, 120], False))
    @unpack
    def testCheckBoundary(self, position, groundTruth):
        self.checkBoundary = CheckBoundary(self.xBoundary, self.yBoundary)
        returnedValue = self.checkBoundary(position)
        truthValue = returnedValue == groundTruth
        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()