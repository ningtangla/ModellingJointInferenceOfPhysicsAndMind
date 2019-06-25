import sys
sys.path.append('..')

import unittest
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.envDiscreteGrid import *
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState


@ddt
class TestEnvironment(unittest.TestCase):
	def setUp(self):
		self.actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]
		self.lowerBoundAngle = 0
		self.upperBoundAngle = np.pi/2
		self.wolfID = 0
		self.sheepID = 1
		self.masterID = 2
		self.positionIndex = [0, 1]

		self.getPredatorPos = GetAgentPosFromState(self.wolfID, self.positionIndex)
		self.getPreyPos = GetAgentPosFromState(self.sheepID, self.positionIndex)
		self.getMasterPos = GetAgentPosFromState(self.masterID, self.positionIndex)

		self.samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors, self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle)

		self.lowerBoundary = 1

		self.adjustingParam = 2
		self.getPullingForceValue = GetPullingForceValue(self.adjustingParam, roundNumber)
		self.getPulledAgentForce = GetPulledAgentForce(self.getMasterPos, self.getPredatorPos, self.samplePulledForceDirection, self.getPullingForceValue)


	@data(
		((-2, -1), {(-1, 0): 0.5, (0, -1): 0.5}),
		((1, 0), {(1, 0): 1}),
		((0, -1), {(0, -1): 1}),
		((0, 1), {(0, 1): 1}),
		((-1, 0), {(-1, 0): 1}),
		((0,0), {(0,0):1}),
		((2, -1), {(1, 0): 0.5, (0, -1): 0.5}),
		((-1, -1), {(-1, 0): 0.5, (0, -1): 0.5}),
		((2, 1), {(1, 0): 0.5, (0, 1): 0.5})
		)
	@unpack
	def testSamplePulledForceDirection(self, pullingDirection, truePulledActionProb):

		iterationTime = 10000
		trueActionLikelihoodPair = zip(truePulledActionProb.keys(), truePulledActionProb.values())

		truePulledActionCount = {action: trueActionProb * iterationTime
		for action, trueActionProb in trueActionLikelihoodPair}

		pulledActionList = [self.samplePulledForceDirection(pullingDirection)
		for _ in range(iterationTime)]

		for action in truePulledActionCount.keys():
			self.assertAlmostEqual(truePulledActionCount[action], pulledActionList.count(action), delta = 200)

	@data(
		((3,4), 4),
		((1,1), 2),
		((1,3), 3),
		((0,1), 2)
	)
	@unpack
	def testPullingForce(self, relativeLocation, trueForce):
		force = self.getPullingForceValue(relativeLocation)
		self.assertEqual(force, trueForce)



	@data(
		([(5,2), (2,3), (3,1)], {(-2,0): 0.5, (0, -2): 0.5}),
		([(2,1), (2,3), (3,3)], {(2,0): 0.5, (0,2): 0.5})
		)
	@unpack
	def testPulledAgentForce(self, state, truePulledActionProb):

		iterationTime = 10000

		truePulledAction = {force: truePulledActionProb[force] * iterationTime for force in truePulledActionProb.keys()}

		forceList = [tuple(self.getPulledAgentForce(state)) for _ in range(iterationTime)]

		for agentForce in truePulledAction.keys():
			self.assertAlmostEqual(truePulledAction[agentForce], forceList.count(agentForce), delta = 200)



	@data(
		((1,1), [(5,2), (2,3), (3,1)],
			{(4, 3): 0.5, (5, 1): 0.5}),
		((1,1), [(2,1), (2,3), (3,3)],
			{(5, 2): 0.5, (3, 4): 0.5})
		)
	@unpack
	def testTransition(self, predatorAction, state, truePredatorNextStateProb):
		stayWithinBoundary = StayWithinBoundary((5,5), 1)
		transitPredator = TransitAgent(stayWithinBoundary, self.getPulledAgentForce,self.getPredatorPos)

		iterationTime = 10000
		predatorNextStateList = [transitPredator(predatorAction, state) for _ in range(iterationTime)]

		truePredatorNextState = {predatorState: truePredatorNextStateProb[predatorState] * iterationTime for predatorState in truePredatorNextStateProb.keys()}

		for nextState in truePredatorNextState.keys():
			self.assertAlmostEqual(truePredatorNextState[nextState], predatorNextStateList.count(nextState), delta = 200)



	@data(((5,5), (1,6), (1,5)),
		((5,5), (-2,2), (1,2)),
		((6,6), (0,7), (1,6)))
	@unpack
	def testBoundaryStaying(self, gridSize, nextIntendedState, trueNextState):
		stayWithinBoundary = StayWithinBoundary(gridSize, self.lowerBoundary)
		nextState = stayWithinBoundary(nextIntendedState)
		self.assertEqual(nextState, trueNextState )


	def tearDown(self):
		pass


if __name__ == "__main__":
	envTest = unittest.TestLoader().loadTestsFromTestCase(TestEnvironment)
	unittest.TextTestRunner(verbosity=2).run(envTest)


