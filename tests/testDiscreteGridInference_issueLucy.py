import sys
import os
import unittest
from ddt import ddt, data, unpack

sys.path.append(os.path.join('..', 'src'))
sys.path.append(os.path.join('..', 'visualize'))

from constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from inferChasing.discreteGridPolicy import UniformPolicy, ActHeatSeeking, \
    HeatSeekingPolicy, WolfPolicy, SheepPolicy, MasterPolicy
from inferChasing.discreteGridTransition import StayWithinBoundary, PulledForceLikelihood, \
    PulledTransition, NoPullTransition
from constrainedChasingEscapingEnv.state import GetAgentPosFromState

import numpy as np

@ddt
class testInference(unittest.TestCase):
    def setUp(self):

        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)

        positionIndex = [0, 1]
        getAgentPosition = lambda agentID, state: GetAgentPosFromState(agentID, positionIndex)(state)

        rationalityParam = 0.9
        heatSeekingPolicy = HeatSeekingPolicy(rationalityParam, self.actHeatSeeking)
        self.wolfPolicy = WolfPolicy(getAgentPosition, heatSeekingPolicy)
        self.sheepPolicy = SheepPolicy(getAgentPosition, heatSeekingPolicy)

        uniformPolicy = UniformPolicy(self.actionSpace)
        self.masterPolicy = MasterPolicy(uniformPolicy)

        self.policyList = [self.wolfPolicy, self.sheepPolicy, self.masterPolicy]
        self.policy = lambda mind, state, allAgentsAction: np.product(
            [agentPolicy(mind, state, allAgentsAction) for agentPolicy in self.policyList])


        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.pulledForceLikelihood = PulledForceLikelihood(self.forceSpace, self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)
        gridSize = (10, 10)
        lowerBoundary = 1
        self.stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)
        self.pulledTransition = PulledTransition(getAgentPosition, self.pulledForceLikelihood, self.stayWithinBoundary)
        self.noPullTransition = NoPullTransition(getAgentPosition, self.stayWithinBoundary)
        transitionList = [self.pulledTransition, self.noPullTransition]
        self.transition = lambda physics, state, allAgentsAction, nextState: \
            np.product([agentTransition(physics, state, allAgentsAction, nextState) for agentTransition in transitionList])


    @data(([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           ("sheep", "wolf", "master"),
           0.1/2* 0.45* 0.25
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           ("master", "sheep", "wolf"),
           0.25* 0.45* 0.45
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           ("sheep", "master", "wolf"),
           0.1/2 * 0.25 * 0.45
           )
        )
    @unpack
    def testPolicy(self, state, allAgentsActions, mind, truePolicyLikelihood):
        policyLikelihood = self.policy(mind, state, allAgentsActions)
        self.assertAlmostEqual(policyLikelihood, truePolicyLikelihood)


    @data(([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (0, 0), (0, 0)),
           ((2, 3), (2, 3), (3, 5)),
           ("noPull", "pulled", "pulled"),
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (0, 0), (0, 0)),
           ((3, 2), (3, 3), (3, 5)),
           ("pulled", "noPull", "pulled"),
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (0, 0), (0, 0)),
           ((2, 3), (4, 3), (3, 5)),
           ("noPull", "pulled", "pulled"),
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (0, 0), (0, 0)),
           ((3, 2), (3, 3), (3, 5)),
           ("noPull", "pulled", "pulled"),
           0
           ),
          ([(2, 2), (3, 3), (2, 2)],
           ((0, 0), (0, 0), (0, 0)),
           ((2, 2), (3, 3), (2, 2)),
           ("pulled", "noPull", "pulled"),
           1
           )
          )
    @unpack
    def testTransition(self, state, allAgentsAction, nextState, pullingIndices, trueTransitionLikelihood):
        transitionLikelihood = self.transition(pullingIndices, state, allAgentsAction, nextState)
        self.assertAlmostEqual(transitionLikelihood, trueTransitionLikelihood)


if __name__ == '__main__':
    unittest.main()
