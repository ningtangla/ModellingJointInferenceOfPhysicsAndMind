import sys

sys.path.append('..')

import unittest
from ddt import ddt, data, unpack

from src.constrainedChasingEscapingEnv.envDiscreteGrid import *
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


@ddt
class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.lowerBoundary = 1

        self.wolfID = 0
        self.sheepID = 1
        self.masterID = 2
        self.positionIndex = [0, 1]

        self.getWolfPos = GetAgentPosFromState(self.wolfID, self.positionIndex)
        self.getSheepPos = GetAgentPosFromState(self.sheepID, self.positionIndex)
        self.getMasterPos = GetAgentPosFromState(self.masterID, self.positionIndex)
        
        self.pulledAgentID = self.wolfID
        self.noPullingAgentID = self.sheepID
        self.pullingAgentID = self.masterID

        self.getPulledAgentPos = GetAgentPosFromState(self.pulledAgentID, self.positionIndex)
        self.getNoPullAgentPos = GetAgentPosFromState(self.noPullingAgentID, self.positionIndex)
        self.getPullingAgentPos = GetAgentPosFromState(self.pullingAgentID, self.positionIndex)
        
        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors,
                                                                     self.forceSpace,
                                                                     self.lowerBoundAngle,
                                                                     self.upperBoundAngle)
        
        self.distanceForceRatio = 2
        self.getPullingForceValue = GetPullingForceValue(self.distanceForceRatio)

        self.getPulledAgentForce = GetPulledAgentForce(self.getPullingAgentPos,
                                                       self.getPulledAgentPos,
                                                       self.samplePulledForceDirection,
                                                       self.getPullingForceValue)

        self.getAgentsForce = GetAgentsForce(self.getPulledAgentForce,
                                             self.pulledAgentID,
                                             self.noPullingAgentID,
                                             self.pullingAgentID)

    @data(
        ((-2, -1), {(-1, 0): 0.5, (0, -1): 0.5}),
        ((1, 0), {(1, 0): 1}),
        ((0, -1), {(0, -1): 1}),
        ((0, 1), {(0, 1): 1}),
        ((-1, 0), {(-1, 0): 1}),
        ((0, 0), {(0, 0): 1}),
        ((2, -1), {(1, 0): 0.5, (0, -1): 0.5}),
        ((2, 1), {(1, 0): 0.5, (0, 1): 0.5})
    )
    @unpack
    def testSamplePulledForceDirection(self, pullersRelativeLocation, truePulledForceProb):

        iterationTime = 10000
        trueForceLikelihoodPair = zip(truePulledForceProb.keys(), truePulledForceProb.values())

        truePulledForceCount = {force: trueForceProb * iterationTime
                                 for force, trueForceProb in trueForceLikelihoodPair}

        pulledForceList = [self.samplePulledForceDirection(pullersRelativeLocation) for _ in range(iterationTime)]

        for force in truePulledForceCount.keys():
            self.assertAlmostEqual(truePulledForceCount[force], pulledForceList.count(force), delta=200)

    @data(
        ((3, 4), 3),
        ((1, 1), 1),
        ((1, 3), 2),
        ((0, 1), 1)
    )
    @unpack
    def testPullingForce(self, pullersRelativeLocation, trueForce):
        force = self.getPullingForceValue(pullersRelativeLocation)
        self.assertEqual(force, trueForce)

    @data(
        ([(5,2), (2,3), (3,1)], {(-2,0): 0.5, (0, -2): 0.5}),
        ([(2,1), (2,3), (3,3)], {(2,0): 0.5, (0,2): 0.5})
        )
    @unpack
    def testPulledAgentForce(self, state, truePulledForceProb):

        iterationTime = 10000

        truePulledForce = {force: truePulledForceProb[force] * iterationTime for force in truePulledForceProb.keys()}

        forceList = [tuple(self.getPulledAgentForce(state)) for _ in range(iterationTime)]

        for agentForce in truePulledForce.keys():
            self.assertAlmostEqual(truePulledForce[agentForce], forceList.count(agentForce), delta = 200)


    @data(
        ([(5, 2), (2, 3), (3, 1)], {(-2, 0): 0.5, (0, -2): 0.5}, {(0, 0): 1}, {(2, 0): 0.5, (0, 2): 0.5}),
        ([(2, 1), (2, 3), (3, 3)], {(2, 0): 0.5, (0, 2): 0.5}, {(0, 0): 1}, {(-2, 0): 0.5, (0, -2): 0.5})
    )
    @unpack
    def testAgentsForce(self, state, truePulledForceProb, trueNoForceProb, truePullingForceProb):

        iterationTime = 10000

        getTrueValue = lambda probDict, iteration: {force: probDict[force]* iteration for force in probDict.keys()}
        pulledTrueForce = getTrueValue(truePulledForceProb, iterationTime)
        noPullTrueForce = getTrueValue(trueNoForceProb, iterationTime)
        pullingTrueForce = getTrueValue(truePullingForceProb, iterationTime)

        forceList = [self.getAgentsForce(state) for _ in range(iterationTime)]

        getAgentForceFromList = lambda agentsForce, agentID: [tuple(forces[agentID]) for forces in agentsForce]
        pulledForce = getAgentForceFromList(forceList, self.pulledAgentID)
        noPullForce = getAgentForceFromList(forceList, self.noPullingAgentID)
        pullingForce = getAgentForceFromList(forceList, self.pullingAgentID)

        for agentForce in pulledTrueForce.keys():
            self.assertAlmostEqual(pulledTrueForce[agentForce], pulledForce.count(agentForce), delta=200)

        for agentForce in noPullTrueForce.keys():
            self.assertAlmostEqual(noPullTrueForce[agentForce], noPullForce.count(agentForce), delta=200)

        for agentForce in pullingTrueForce.keys():
            self.assertAlmostEqual(pullingTrueForce[agentForce], pullingForce.count(agentForce), delta=200)

    @data(
        ([(1, 1), (1, 1), (1, 1)], [(5, 2), (2, 3), (3, 1)],
         {(4, 3): 0.5, (5, 1): 0.5}),
        ([(1, 1), (1, 1), (1, 1)], [(2, 1), (2, 3), (3, 3)],
         {(5, 2): 0.5, (3, 4): 0.5})
    )
    @unpack
    def testTransition(self, allActions, state, truePulledAgentNextStateProb):
        stayWithinBoundary = StayWithinBoundary((5, 5), 1)
        transition = Transition(stayWithinBoundary, self.getAgentsForce)

        iterationTime = 10000
        pulledNextStateList = [transition(allActions, state)[self.pulledAgentID] for _ in range(iterationTime)]

        pulledTrueNextState = {pulledState: truePulledAgentNextStateProb[pulledState] * iterationTime for pulledState in truePulledAgentNextStateProb.keys()}

        for nextState in pulledTrueNextState.keys():
            self.assertAlmostEqual(pulledTrueNextState[nextState], pulledNextStateList.count(nextState), delta=200)



    @data(((5, 5), (1, 6), (1, 5)),
          ((5, 5), (-2, 2), (1, 2)),
          ((6, 6), (0, 7), (1, 6)))
    @unpack
    def testBoundaryStaying(self, gridSize, nextIntendedState, trueNextState):
        stayWithinBoundary = StayWithinBoundary(gridSize, self.lowerBoundary)
        nextState = stayWithinBoundary(nextIntendedState)
        self.assertEqual(nextState, trueNextState)

    def tearDown(self):
        pass


@ddt
class TestEnvironmentRandomizedCase(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi / 2
        self.lowerBoundary = 1

        self.wolfID = 2
        self.sheepID = 0
        self.masterID = 1
        self.positionIndex = [0, 1]

        self.getWolfPos = GetAgentPosFromState(self.wolfID, self.positionIndex)
        self.getSheepPos = GetAgentPosFromState(self.sheepID, self.positionIndex)
        self.getMasterPos = GetAgentPosFromState(self.masterID, self.positionIndex)

        self.pulledAgentID = self.wolfID
        self.noPullingAgentID = self.sheepID
        self.pullingAgentID = self.masterID

        self.getPulledAgentPos = GetAgentPosFromState(self.pulledAgentID, self.positionIndex)
        self.getNoPullAgentPos = GetAgentPosFromState(self.noPullingAgentID, self.positionIndex)
        self.getPullingAgentPos = GetAgentPosFromState(self.pullingAgentID, self.positionIndex)

        self.forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.samplePulledForceDirection = SamplePulledForceDirection(
            computeAngleBetweenVectors, self.forceSpace, self.lowerBoundAngle,
            self.upperBoundAngle)

        self.distanceForceRatio = 2
        self.getPullingForceValue = GetPullingForceValue(
            self.distanceForceRatio)

        self.getPulledAgentForce = GetPulledAgentForce(self.getPullingAgentPos,
                                                       self.getPulledAgentPos,
                                                       self.samplePulledForceDirection,
                                                       self.getPullingForceValue)

        self.getAgentsForce = GetAgentsForce(self.getPulledAgentForce,
                                             self.pulledAgentID,
                                             self.noPullingAgentID,
                                             self.pullingAgentID)

    @data(
        ([(5, 2), (2, 3), (3, 1)], {(-2, 0): 0.5, (0, 2): 0.5}),
        ([(2, 1), (2, 3), (3, 3)], {(-1, 0): 1}),
        ([(5, 5), (1, 1), (4, 4)], {(-3, 0): 0.5, (0, -3): 0.5}),
        ([(6, 3), (2, 4), (8, 8)], {(-4, 0): 0.5, (0, -4): 0.5})
    )
    @unpack
    def testPulledAgentForce(self, state, truePulledForceProb):

        iterationTime = 10000
        truePulledForce = {force: truePulledForceProb[force] * iterationTime for force in truePulledForceProb.keys()}

        forceList = [tuple(self.getPulledAgentForce(state)) for _ in range(iterationTime)]

        for agentForce in truePulledForce.keys():
            self.assertAlmostEqual(truePulledForce[agentForce], forceList.count(agentForce), delta=200)


    @data(
        ([(2, 3), (3, 1), (2, 2)], {(1, 0): 0.5, (0, -1): 0.5}, {(0, 0): 1}, {(-1, 0): 0.5, (0, 1): 0.5}),
        ([(2, 3), (3, 3), (2, 1)], {(2, 0): 0.5, (0, 2): 0.5}, {(0, 0): 1}, {(-2, 0): 0.5, (0, -2): 0.5})
    )
    @unpack
    def testAgentsForce(self, state, truePulledForceProb, trueNoForceProb, truePullingForceProb):

        iterationTime = 10000

        getTrueValue = lambda probDict, iteration: {force: probDict[force] * iteration for force in probDict.keys()}
        pulledTrueForce = getTrueValue(truePulledForceProb, iterationTime)
        noPullTrueForce = getTrueValue(trueNoForceProb, iterationTime)
        pullingTrueForce = getTrueValue(truePullingForceProb, iterationTime)

        forceList = [self.getAgentsForce(state) for _ in range(iterationTime)]

        getAgentForceFromList = lambda agentsForce, agentID: [tuple(forces[agentID]) for forces in agentsForce]
        pulledForce = getAgentForceFromList(forceList, self.pulledAgentID)
        noPullForce = getAgentForceFromList(forceList, self.noPullingAgentID)
        pullingForce = getAgentForceFromList(forceList, self.pullingAgentID)

        for agentForce in pulledTrueForce.keys():
            self.assertAlmostEqual(pulledTrueForce[agentForce], pulledForce.count(agentForce), delta=200)

        for agentForce in noPullTrueForce.keys():
            self.assertAlmostEqual(noPullTrueForce[agentForce], noPullForce.count(agentForce), delta=200)

        for agentForce in pullingTrueForce.keys():
            self.assertAlmostEqual(pullingTrueForce[agentForce], pullingForce.count(agentForce), delta=200)


    @data(
        ([(1, 1), (1, 1), (1, 1)], [(5, 2), (2, 3), (3, 1)],
         {(2, 2): 0.5, (4, 4): 0.5}),
        ([(1, 1), (1, 1), (1, 1)], [(2, 1), (2, 3), (3, 3)],
         {(3, 4): 1}),
        ([(0, -1), (1, 0), (-1, 0)],[(6, 3), (2, 4), (8, 8)],
         {(3,8): 0.5, (7,4): 0.5})
    )
    @unpack
    def testTransition(self, allActions, state, truePulledAgentNextStateProb):
        stayWithinBoundary = StayWithinBoundary((10, 10), 1)
        transition = Transition(stayWithinBoundary, self.getAgentsForce)

        iterationTime = 10000
        pulledNextStateList = [transition(allActions, state)[self.pulledAgentID] for _ in range(iterationTime)]

        pulledTrueNextState = {pulledState: truePulledAgentNextStateProb[pulledState] * iterationTime for pulledState in truePulledAgentNextStateProb.keys()}

        for nextState in pulledTrueNextState.keys():
            self.assertAlmostEqual(pulledTrueNextState[nextState], pulledNextStateList.count(nextState), delta=200)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()


