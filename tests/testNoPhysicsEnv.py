import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.constrainedChasingEscapingEnv.envNoPhysics import RandomReset, TransiteForNoPhysics, IsTerminal, IsTerminalWithInterpolation,StayInBoundaryByReflectVelocity, CheckBoundary,UnpackCenterControlAction,TransiteCenterControlActionForNoPhysics
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
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
        self.minDistanceInterpolation = 1.414
        self.divideDegree=5
        self.centerControlIndex=[1]
        self.getPreyPos = GetAgentPosFromState(
            self.sheepId, self.posIndex)
        self.getPredatorPos = GetAgentPosFromState(
            self.wolfId, self.posIndex)


        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)
        self.isTerminal = IsTerminal(
            self.getPredatorPos, self.getPreyPos, self.minDistance)
        self.isTerminalWithInterpolation = IsTerminalWithInterpolation(self.getPredatorPos, self.getPreyPos, self.minDistanceInterpolation,self.divideDegree)
        self.transition = TransiteForNoPhysics(self.stayInBoundaryByReflectVelocity)
        self.unpackAction=UnpackCenterControlAction(self.centerControlIndex)
        self.transitionForActionControl=TransiteCenterControlActionForNoPhysics(self.stayInBoundaryByReflectVelocity,self.unpackAction)
    @data((np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])), (np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[2, 2], [3, 5]])))
    @unpack
    def testTransition(self, state, action, groundTruthReturnedNextState):
        nextState = self.transition(state, action)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())
    
    @data(( np.array([[0, 0],[[0, 0],[0, 0]]]),np.array([[0, 0],[0, 0],[0, 0]]) ), ( np.array([[1, 0], [[0, 1],[1, 0]]]), np.array([[1, 0], [0, 1],[1, 0]])))
    @unpack
    def testUnpackCenterControlAction(self,centerControlAction,graoudTruthUnpackedAction):
        upackedAction=self.unpackAction(centerControlAction)
        truthValue = upackedAction == graoudTruthUnpackedAction
        self.assertTrue(truthValue.all())

    @data((np.array([[0, 0], [0, 0], [0, 0]]), np.array([[0, 0], [[0, 0],[0, 0]]]), np.array([[0, 0], [0, 0], [0, 0]])), (np.array([[1, 2], [3, 4], [3, 4]]), np.array([[1, 0], [[0, 1],[1, 0]]]), np.array([[2, 2], [3, 5], [4, 4]])))
    @unpack
    def testTransiteCenterControlActionForNoPhysics(self, state, action, groundTruthReturnedNextState):
        nextState = self.transitionForActionControl(state, action)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())


    @data(([[2, 2], [10, 10]], True), ([[10, 23], [100, 100]], False))
    @unpack
    def testIsTerminal(self, state, groundTruthTerminal):
        terminal = self.isTerminal(state)
        self.assertEqual(terminal, groundTruthTerminal)

    @data(([[0, 0], [6, 0]],[[6, 6], [0, 6]], True), ([[0, 0], [6, 0]],[[12, 12], [0, 6]], False))
    @unpack
    def testIsTerminalwithInterpolation(self,lastState, currentState, groundTruthTerminal):
        terminal = self.isTerminalWithInterpolation(lastState, currentState)
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
