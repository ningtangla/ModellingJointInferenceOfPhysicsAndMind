import sys
sys.path.append('../src/sheepWolf')

import unittest
from ddt import ddt, data, unpack

from envDiscreteGrid import *
from calculateAngleFunction import *
from discreteGridWrapperFunctions import LocateAgent


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

		self.locateWolf = LocateAgent(self.wolfID, self.positionIndex)
		self.locateSheep = LocateAgent(self.sheepID, self.positionIndex)
		self.locateMaster = LocateAgent(self.masterID, self.positionIndex)

		self.samplePulledForceDirection = SamplePulledForceDirection(calculateAngle, self.actionSpace, self.lowerBoundAngle, self.upperBoundAngle)

		self.lowerBoundary = 1

		self.adjustingParam = 2
		self.getPullingForceValue = GetPullingForceValue(self.adjustingParam, roundNumber)

		self.getAgentsForceAction = GetAgentsForceAction(self.locateMaster, self.locateWolf, self.samplePulledForceDirection, self.getPullingForceValue)

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
		([(5,2), (2,3), (3,1)], {(-2,0): 0.5, (0, -2): 0.5}, {(0,0): 1}, {(2,0): 0.5, (0,2): 0.5}),
		([(2,1), (2,3), (3,3)], {(2,0): 0.5, (0,2): 0.5}, {(0,0): 1}, {(-2,0): 0.5, (0, -2): 0.5})
		)
	@unpack
	def testAgentsForceAction(self, state, wolfForceProb, sheepForceProb, masterForceProb):

		iterationTime = 10000

		wolfTrueForce = {force: wolfForceProb[force] * iterationTime for force in wolfForceProb.keys() }
		sheepTrueForce = {force: sheepForceProb[force] * iterationTime for force in sheepForceProb.keys() }
		masterTrueForce = {force: masterForceProb[force] * iterationTime for force in masterForceProb.keys()}

		forceList = [self.getAgentsForceAction(state) for _ in range(iterationTime)]
		wolfForce = [tuple(force[0]) for force in forceList]
		sheepForce = [tuple(force[1]) for force in forceList]
		masterForce = [tuple(force[2]) for force in forceList]


		for agentForce in wolfTrueForce.keys():
			self.assertAlmostEqual(wolfTrueForce[agentForce], wolfForce.count(agentForce), delta = 200)
		
		for agentForce in sheepTrueForce.keys():
			self.assertAlmostEqual(sheepTrueForce[agentForce], sheepForce.count(agentForce), delta = 200)
		
		for agentForce in masterTrueForce.keys():
			self.assertAlmostEqual(masterTrueForce[agentForce], masterForce.count(agentForce), delta = 200)
	



	@data(
		([(1,1), (1,1), (1,1)], [(5,2), (2,3), (3,1)], 
			{(4,3): 0.5, (5, 1): 0.5}),
		([(1,1), (1,1), (1,1)], [(2,1), (2,3), (3,3)],
			{(5,2): 0.5, (3, 4): 0.5})
		)
	@unpack
	def testTransition(self, allActions, state, trueWolfNextStateProb):
		stayWithinBoundary = StayWithinBoundary((5,5), 1)
		transition = Transition(stayWithinBoundary, self.getAgentsForceAction)

		iterationTime = 10000
		wolfNextStateList = [transition(allActions, state)[0] for _ in range(iterationTime)]

		wolfTrueNextState = {wolfState: trueWolfNextStateProb[wolfState] * iterationTime for wolfState in trueWolfNextStateProb.keys() }

		for nextState in wolfTrueNextState.keys():
			self.assertAlmostEqual(wolfTrueNextState[nextState], wolfNextStateList.count(nextState), delta = 200)



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


