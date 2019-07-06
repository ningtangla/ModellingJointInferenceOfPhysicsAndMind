import sys
import os
import unittest
from ddt import ddt, data, unpack

sys.path.append(os.path.join('..', 'src', 'inferDiscreteGridChasing'))
sys.path.append(os.path.join('..', 'src', 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join('..', 'visualize'))

from analyticGeometryFunctions import computeAngleBetweenVectors
from policyLikelihood import UniformPolicy, ActHeatSeeking, \
    HeatSeekingActionLikelihood, WolfPolicy, SheepPolicy, MasterPolicy
from transitionLikelihood import StayWithinBoundary, PulledForceLikelihood, \
    PulledTransition, NoPullTransition
from state import GetAgentPosFromState

import numpy as np

@ddt
class testInference(unittest.TestCase):
    def setUp(self):

        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)

        positionIndex = [0, 1]
        self.getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)
        rationalityParam = 0.9
        self.getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam, self.actHeatSeeking, getWolfPos, getSheepPos)
        self.wolfPolicy = WolfPolicy(self.getAgentPosition, self.getHeatSeekingActionLikelihood)
        self.sheepPolicy = SheepPolicy(self.getAgentPosition, self.getHeatSeekingActionLikelihood)

        self.uniformPolicy = UniformPolicy(self.actionSpace)
        self.masterPolicy = MasterPolicy(self.uniformPolicy)

        self.policyList = [self.wolfPolicy, self.sheepPolicy, self.masterPolicy]
        self.policy = lambda mind, state, allAgentsAction: np.product(
            [agentPolicy(mind, state, allAgentsAction) for agentPolicy in self.policyList])


        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.pulledForceLikelihood = PulledForceLikelihood(self.forceSpace, self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)
        gridSize = (10, 10)
        lowerBoundary = 1
        self.stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)
        self.pulledTransition = PulledTransition(self.getAgentPosition, self.pulledForceLikelihood, self.stayWithinBoundary)
        self.noPullTransition = NoPullTransition(self.getAgentPosition, self.stayWithinBoundary)
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
