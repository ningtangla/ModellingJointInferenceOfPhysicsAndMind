import sys
import unittest
from ddt import ddt, data, unpack

sys.path.append('..')
sys.path.append('../src/inferDiscreteGridChasing')
sys.path.append('../src/constrainedChasingEscapingEnv')

from forceLikelihood import *
from heatSeekingLikelihood import *
from analyticGeometryFunctions import computeAngleBetweenVectors
from inference import *
from wrapperFunctions import *

@ddt
class testInference(unittest.TestCase):
    def setUp(self):
        self.chasingAgents = [0, 1, 2]
        self.pullingAgents = [0, 1, 0]
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]

        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.getWolfProperAction = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)
        self.getSheepProperAction = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)

        positionIndex = [0, 1]
        rationalityParam = 0.9

        self.getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)

        self.getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam,self.getWolfProperAction, getWolfPos, getSheepPos)
        self.getMasterActionLikelihood = RandomActionLikelihood(self.actionSpace)

        self.getPolicyDistribution = GetPolicyDistribution(self.getAgentPosition, self.getHeatSeekingActionLikelihood,self.getMasterActionLikelihood)
        self.inferPolicyLikelihood = InferPolicyLikelihood(self.getPolicyDistribution)

        self.getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(computeAngleBetweenVectors, self.forceSpace,self.lowerBoundAngle, self.upperBoundAngle)
        self.getPulledAgentForceDistribution = GetPulledAgentForceDistribution(self.getAgentPosition, self.getPulledAgentForceLikelihood)

        self.inferForceLikelihood = InferForceLikelihood(self.getPulledAgentForceDistribution)


    @data(([(2,2), (3,3), (4,5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 0, 2),
           0.1/2* 0.45* 0.25
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (2, 1, 0), #master sheep wolf
           0.25* 0.45* 0.45
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 2, 0),  # sheep master wolf
           0.1/2 * 0.25 * 0.45
           )
        )
    @unpack
    def testPolicyInference(self, state, allAgentsActions, chasingIndices, truePolicyLikelihood):
        policyLikelihood = self.inferPolicyLikelihood(state, allAgentsActions, chasingIndices)
        self.assertAlmostEqual(policyLikelihood, truePolicyLikelihood)


    @data(([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 0, 0), #noPull, pulled, puller
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((1, 0), (0, 0), (-1, 0)),
           (0, 1, 0),  #pulling, noPull, pulled
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (1, 0), (-1, 0)),
           (1, 0, 0),  # noPull, pulling, pulled
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (-1, 0), (1, 0)),
           (1, 0, 0),  # noPull, pulling, pulled
           0
           ),
          ([(2, 2), (3, 3), (2, 2)],
           ((0, 0), (0, 0), (0, 0)),
           (0, 1, 0),  # pulling, noPull, pulled
           1
           )
          )
    @unpack
    def testForceInference(self, state, allAgentsForce, pullingIndices, trueForceLikelihood):
        forceLikelihood = self.inferForceLikelihood(state, allAgentsForce, pullingIndices)
        self.assertAlmostEqual(forceLikelihood, trueForceLikelihood)

    @data(
        ([(1, 1), (1, 1), (1, 1)], [(1, 1), (1, 1), (1, 1)], 1),
        ([(1, 2), (1, 1), (1, 1)], [(1, 1), (1, 1), (1, 1)], 0)
    )
    @unpack
    def testTransitionInference(self, expectedNextState, observedNextState, trueLikelihood):
        likelihood = inferTransitionLikelihood(expectedNextState, observedNextState)
        self.assertEqual(likelihood, trueLikelihood)


    @data
    @unpack
    def checkIndex(self, index):
        self.index = createIndex(self.chasingAgents, self.pullingAgents, self.actionSpace, self.forceSpace)
        self.assertEqual(len(self.index), 13* 64* 36)

if __name__ == '__main__':
    unittest.main()
