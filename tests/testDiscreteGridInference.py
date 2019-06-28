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
        self.pullingAgents = [0, 1, 2]
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.index = createIndex(self.chasingAgents, self.pullingAgents, self.actionSpace, self.forceSpace)

        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.getWolfProperAction = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)
        self.getSheepProperAction = ActHeatSeeking(self.actionSpace, computeAngleBetweenVectors, self.lowerBoundAngle, self.upperBoundAngle)

        positionIndex = [0, 1]
        rationalityParam = 0.9

        self.getWolfPosition = lambda wolfID: GetAgentPosFromState(wolfID, positionIndex)
        self.getSheepPosition = lambda sheepID: GetAgentPosFromState(sheepID, positionIndex)

        self.getWolfActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam,self.getWolfProperAction, getWolfPos, getSheepPos)
        self.getSheepActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam,self.getSheepProperAction,getWolfPos, getSheepPos)
        self.getMasterActionLikelihood = RandomActionLikelihood(self.actionSpace)

        self.inferPolicyLikelihood = InferPolicyLikelihood(positionIndex, rationalityParam, self.actionSpace, self.getWolfPosition, self.getSheepPosition,self.getWolfActionLikelihood, self.getSheepActionLikelihood,self.getMasterActionLikelihood)

        self.getPulledAgentPos = lambda pulledAgentID: GetAgentPosFromState(pulledAgentID, positionIndex)
        self.getPullingAgentPos = lambda pullingAgentID: GetAgentPosFromState(pullingAgentID, positionIndex)

        self.getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(computeAngleBetweenVectors, self.forceSpace,self.lowerBoundAngle, self.upperBoundAngle)

        self.inferForceLikelihood = InferForceLikelihood(positionIndex, self.forceSpace, self.getPulledAgentPos, self.getPullingAgentPos, self.getPulledAgentForceLikelihood)

    @data(([(2,2), (3,3), (4,5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 0, 2),
           0.1/3* 0.45*0.2
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (2, 1, 0), #master sheep wolf
           0.2* 0.45* 0.45
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 2, 0),  # sheep master wolf
           0.1/3 * 0.2 * 0.45
           )
        )
    @unpack
    def testPolicyInference(self, state, allAgentsActions, chasingIndices, truePolicyLikelihood):
        policyLikelihood = self.inferPolicyLikelihood(state, allAgentsActions, chasingIndices)
        self.assertAlmostEqual(policyLikelihood, truePolicyLikelihood)


    @data(([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0), (-1, 0)),
           (1, 0, 2), #noPull, pulled, puller
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((1, 0), (0, 0), (-1, 0)),
           (2, 1, 0),  #pulling, noPull, pulled
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (-1, 0), (-1, 0)),
           (1, 2, 0),  #noPull, pulling, pulled
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (1, 0), (-1, 0)),
           (1, 2, 0),  # noPull, pulling, pulled
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (-1, 0), (1, 0)),
           (1, 2, 0),  # noPull, pulling, pulled
           0
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


if __name__ == '__main__':
    unittest.main()
