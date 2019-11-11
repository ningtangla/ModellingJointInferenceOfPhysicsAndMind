import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Local import

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

from exec.evaluateSupervisedLearning.trainSheepInSingleChasingNoPhysics.sampleMCTSSheepSingleChasingNoPhysics import IsTerminal
@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.wolfId = 1
        self.posIndex = [0, 1]
        self.minDistance = 1.414
        self.divideDegree=5
        self.getPreyPos = GetAgentPosFromState(
            self.sheepId, self.posIndex)
        self.getPredatorPos = GetAgentPosFromState(
            self.wolfId, self.posIndex)

        self.isTerminal = IsTerminal(
            self.getPredatorPos, self.getPreyPos, self.minDistance,self.divideDegree)

    @data(([[0, 0], [6, 0]],[[6, 6], [0, 6]], True), ([[0, 0], [6, 0]],[[12, 12], [0, 6]], False))
    @unpack
    def testIsTerminal(self,lastState, currentState, groundTruthTerminal):
        terminal = self.isTerminal(lastState, currentState)
        self.assertEqual(terminal, groundTruthTerminal)


if __name__ == '__main__':
    unittest.main()
