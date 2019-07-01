import sys
sys.path.append('..')

import unittest
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.envDiscreteGrid import *
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState


@ddt
class TestEnvironment(unittest.TestCase):
	def setUp(self):
		self.actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1),(0, 0)]
		self.lowerBoundAngle = 0
		self.upperBoundAngle = np.pi/2

		self.wolfID = 0
		self.sheepID = 1
		self.masterID = 2
		self.positionIndex = [0, 1]

		self.locateWolf = GetAgentPosFromState(self.wolfID, self.positionIndex)
		self.locateSheep = GetAgentPosFromState(self.sheepID, self.positionIndex)
		self.locateMaster = GetAgentPosFromState(self.masterID, self.positionIndex)

		self.samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors, self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle)

		self.lowerBoundary = 1

		self.adjustingParam = 2
		self.getPullingForce = GetPullingForce(self.adjustingParam, roundNumber)

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


# param = 2
# force = math.ceil(distance / self.adjustingParam + 1)

	@data(
		((3,4), 4),
		((1,1), 2),
		((1,3), 3),
		((0,1), 2)
	)
	@unpack
	def testPullingForce(self, relativeLocation, trueForce):
		force = self.getPullingForce(relativeLocation)
		self.assertEqual(force, trueForce)


	
	@data(
		((5, 5), (1, 0), [(5,2), (2,3), (3,1)], {(5,1): 0.5, (4,2): 0.5}),
		((6, 6), (0, -1), [(2,1), (2,3), (3,3)], {(4,1): 0.5, (2,2): 0.5}),
		((5, 5), (1, 0), [(4, 2), (2,3), (5,5)], {(5,2): 0.5, (5,5): 0.5}),
	
		((5, 5), (1, 0), [(5, 1), (2, 3), (3, 1)], {(4, 1): 1}),
		((6, 6), (0,-1), [(2, 1), (2, 3), (3, 1)], {(4, 1):1}),
		((5, 5), (0, 1), [(5, 4), (4, 5), (5, 5)], {(5, 5):1})
	)
	@unpack
	def testPulledAgentTransition(self, gridSize, action, state, trueNextStateProb):
		stayWithinBoundary = StayWithinBoundary(gridSize, self.lowerBoundary)
	
		getWolfTransition = PulledAgentTransition(stayWithinBoundary, self.samplePulledForceDirection, self.locateMaster, self.locateWolf, self.getPullingForce)
	
		iterationTime = 10000
		trueNextStateProbPair = zip(trueNextStateProb.keys(), trueNextStateProb.values())
		trueNextStateCount = {action: trueNextStateProb * iterationTime
		for action, trueNextStateProb in trueNextStateProbPair}
		nextStateList = [getWolfTransition(action, state) for _ in range(iterationTime)]
		for action in trueNextStateCount.keys():
			self.assertAlmostEqual(trueNextStateCount[action], nextStateList.count(action), delta = 200)
	

	@data(
		((5,5), (1,0), [(2,3),(5,1),(3,1)], (5,1)),
		((6, 6), (0, -1), [(2,3), (2,1), (3,1)] , (2, 1)),
		((3, 3), (1, 0), [(2,3),(2, 2), (3,1)], (3, 2))
		)
	@unpack
	def testPlainTransition(self, gridSize, action, state, trueNextState):
		stayWithinBoundary = StayWithinBoundary(gridSize, self.lowerBoundary)

		transitPlainAgent = PlainTransition(stayWithinBoundary, self.locateSheep)
		nextState = transitPlainAgent(action, state)
		self.assertEqual(nextState, trueNextState)


	@data(((5,5), (1,6), (1,5)),
		((5,5), (-2,2), (1,2)),
		((6,6), (0,7), (1,6)))
	@unpack
	def testBoundaryTransition(self, gridSize, nextIntendedState, trueNextState):
		stayWithinBoundary = StayWithinBoundary(gridSize, self.lowerBoundary)
		nextState = stayWithinBoundary(nextIntendedState)
		self.assertEqual(nextState, trueNextState )


	def tearDown(self):
		pass


if __name__ == "__main__":
    envTest = unittest.TestLoader().loadTestsFromTestCase(TestEnvironment)
    unittest.TextTestRunner(verbosity=2).run(envTest)


