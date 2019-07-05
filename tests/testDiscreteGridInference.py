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

        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)

        positionIndex = [0, 1]
        self.getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)
        rationalityParam = 0.9
        self.getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam, self.actHeatSeeking, getWolfPos, getSheepPos)
        self.getWolfActionProb = GetWolfActionProb(self.getAgentPosition, self.getHeatSeekingActionLikelihood)
        self.getSheepActionProb = GetSheepActionProb(self.getAgentPosition, self.getHeatSeekingActionLikelihood)

        self.getRandomActionLikelihood = GetRandomActionLikelihood(self.actionSpace)
        self.getMasterActionProb = GetMasterActionProb(self.getRandomActionLikelihood)

        self.getAgentsActionProb = [self.getWolfActionProb, self.getSheepActionProb, self.getMasterActionProb]

        self.getPolicyLikelihood = lambda chasingIndices, state, allAgentsAction: np.product(
            [getActionProb(chasingIndices, state, allAgentsAction) for getActionProb in self.getAgentsActionProb])


        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(self.forceSpace,self.lowerBoundAngle, self.upperBoundAngle, computeAngleBetweenVectors)
        self.getPulledAgentForceProb = GetPulledAgentForceProb(self.getAgentPosition, self.getPulledAgentForceLikelihood)

        gridSize = (10, 10)
        lowerBoundary = 1
        self.stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)
        self.getTransitionLikelihood = GetTransitionLikelihood(self.getPulledAgentForceProb, getNoPullAgentForceProb, self.stayWithinBoundary)


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
    def testPolicyLikelihood(self, state, allAgentsActions, chasingIndices, truePolicyLikelihood):
        policyLikelihood = self.getPolicyLikelihood(chasingIndices, state, allAgentsActions)
        self.assertAlmostEqual(policyLikelihood, truePolicyLikelihood)


    @data(([(2, 2), (3, 3), (4, 5)],
           ((0, 1), (-1, 0),(-1, 0)),
           ((0, 0), (0, 0), (0, 0)),
           ((2, 3), (2, 3), (3, 5)),
           ("noPull", "pulled", "pulled"), #noPull, pulled, puller
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((1, 0), (0, 0), (-1, 0)),
           ((0, 0), (0, 0), (0, 0)),
           ((3, 2), (3, 3), (3, 5)),
           ("pulled", "noPull", "pulled"),  #pulling, noPull, pulled
           0.5
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (1, 0), (-1, 0)),
           ((0, 0), (0, 0), (0, 0)),
           ((2, 3), (4, 3), (3, 5)),
           ("noPull", "pulled", "pulled"),  # noPull, pulling, pulled
           0
           ),
          ([(2, 2), (3, 3), (4, 5)],
           ((0, 0), (-1, 0), (1, 0)),
           ((0, 0), (0, 0), (0, 0)),
           ((3, 2), (3, 3), (3, 5)),
           ("noPull", "pulled", "pulled"),  # noPull, pulling, pulled
           0
           ),
          ([(2, 2), (3, 3), (2, 2)],
           ((0, 0), (0, 0), (0, 0)),
           ((0, 0), (0, 0), (0, 0)),
           ((2, 2), (3, 3), (2, 2)),
           ("pulled", "noPull", "pulled"),  # pulling, noPull, pulled
           1
           )
          )
    @unpack
    def testTransitionLikelihood(self, state, allAgentsForce, allAgentsAction, nextState, pullingIndices, trueTransitionLikelihood):
        transitionLikelihood = self.getTransitionLikelihood(pullingIndices, allAgentsForce, allAgentsAction, state, nextState)
        self.assertAlmostEqual(transitionLikelihood, trueTransitionLikelihood)


    def testIndex(self):
        chasingAgents = ['wolf', 'sheep', 'master']
        pullingAgents = ['pulled', 'noPull', 'pulled']
        self.index = createIndex(chasingAgents, pullingAgents, self.actionSpace, self.forceSpace)
        self.assertEqual(len(self.index), 13* 64* 6* 3)

if __name__ == '__main__':
    unittest.main()
