import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append('..')

from exec.evaluateEffectOfVelocityOnValueEstimationChaseMujoco.evaluateEffectOfVelocityOnValueEstimationChaseMujoco\
    import GenerateInitQPosTwoAgentsGivenDistance, GetInitStateFromWolfVelocityDirection
from src.constrainedChasingEscapingEnv.envMujoco import WithinBounds
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeVectorNorm

import unittest
from ddt import ddt, data, unpack
import numpy as np

@ddt
class TestEvaluateEffectOfVelocityOnValueEstimationChaseMujoco(unittest.TestCase):
    def setUp(self):
        self.withinBounds = WithinBounds((-9.7, -9.7, -9.7, -9.7), (9.7, 9.7, 9.7, 9.7))
        self.computeVectorNorm = computeVectorNorm
        self.sheepQPosIndex = [0, 1]
        self.wolfQPosIndex = [2, 3]

    # @data(([2]), ([4]), ([8]), ([16]))
    # @unpack
    # def testGenerateInitQPosTwoAgentsGivenDistance(self, initDistance):
    #     numSamples = 10000
    #     generateInitQPosTwoAgentsGivenDistance = GenerateInitQPosTwoAgentsGivenDistance(-9.7, 9.7, self.withinBounds)
    #     for sample in range(numSamples):
    #         initQPos = generateInitQPosTwoAgentsGivenDistance(initDistance)
    #         actualDistance = self.computeVectorNorm(initQPos[:2]-initQPos[2:4])
    #         self.assertAlmostEqual(initDistance, actualDistance)


    @data(((0, 0, 3, 4), 'towards', 5, np.asarray([[0, 0, 0, 0, 0, 0], [3, 4, 3, 4, -3, -4]])),
          ((0, 0, 3, 4), 'away', 5, np.asarray([[0, 0, 0, 0, 0, 0], [3, 4, 3, 4, 3, 4]])),
          ((0, 0, 3, 4), 'stationary', 5, np.asarray([[0, 0, 0, 0, 0, 0], [3, 4, 3, 4, 0, 0]])))
    @unpack
    def testGetInitStateFromWolfVelocityDirection(self, initQPos, initVelocityDirection, wolfVelocityMagnitude, groundTruthState):
        getInitStateFromWolfVelocityDirection = \
            GetInitStateFromWolfVelocityDirection(self.sheepQPosIndex, self.wolfQPosIndex, self.computeVectorNorm, wolfVelocityMagnitude)
        initState = getInitStateFromWolfVelocityDirection(initQPos, initVelocityDirection)
        self.assertTrue(np.array_equal(initState, groundTruthState))


if __name__ == '__main__':
    unittest.main()