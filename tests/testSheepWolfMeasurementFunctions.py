import sys
import os
sys.path.append(os.path.join('..', 'src', 'sheepWolf'))

import unittest
from ddt import ddt, unpack, data
import numpy as np

from measurementFunctions import computeDistance, ComputeOptimalNextPos, DistanceBetweenActualAndOptimalNextPosition
from sheepWolfWrapperFunctions import GetStateFromTrajectory, GetAgentPosFromState, GetAgentPosFromTrajectory
from policiesFixed import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from envMujoco import TransitionFunction, IsTerminal

@ddt
class TestSheepWolfMeasurementFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(self.actionSpace, self.getWolfXPos, self.getSheepXPos)
        self.killzoneRadius = 0.5
        self.isTerminal = IsTerminal(self.killzoneRadius, self.getSheepXPos, self.getWolfXPos)
        self.modelName = 'twoAgents'
        self.renderOn = False
        self.numSimulationFrames = 20
        self.transit = TransitionFunction(self.modelName, self.isTerminal, self.renderOn, self.numSimulationFrames)

        self.sheepTransit = lambda state, action: self.transit(state, [action, stationaryAgentPolicy(state)])
        self.stateIndex = 0
        self.getInitStateFromTrajectory = GetStateFromTrajectory(0, self.stateIndex)
        self.computeOptimalNextPos = ComputeOptimalNextPos(self.getInitStateFromTrajectory, self.optimalPolicy,
                                                           self.sheepTransit, self.getSheepXPos)

    @data((np.asarray([1, 2]), np.asarray([3, 4]), 2*np.sqrt(2)),
          (np.asarray([-10, 10]), np.asarray([10, -10]), 20*np.sqrt(2)))
    @unpack
    def testComputeDistance(self, pos1, pos2, groundTruthDistance):
        distance = computeDistance(pos1, pos2)

        self.assertEqual(groundTruthDistance, distance)


    @data(([(np.asarray([[3, 4, 3, 4, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))])],
           np.asarray((-7, 7))),
          ([(np.asarray([[3, 3, 3, 3, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
            (np.asarray([[0, 0, 0, 0, 0, 0], [-6, 8, -6, 8, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))])],
           np.asarray((7, 7))))
    @unpack
    def testComputeOptimalNextPos(self, trajectory, optimalAction):
        optimalNextPos = self.computeOptimalNextPos(trajectory)
        getSheepInitPosFromTrajectory = GetAgentPosFromTrajectory(self.getSheepXPos, self.getInitStateFromTrajectory)
        displacement = optimalNextPos - getSheepInitPosFromTrajectory(trajectory)
        hadamardProductDisplacementAndOptimalAction = np.multiply(np.asarray(displacement), np.asarray(optimalAction))

        truthValue = all(i > 0 for i in hadamardProductDisplacementAndOptimalAction)
        self.assertTrue(truthValue)


    @data(([(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))]),
            (np.asarray([[3.2, 4, 3.2, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))])],
           [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))]),
            (np.asarray([[3.1, 4.1, 3.1, 4.1, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))])],
          [(np.asarray([[3, 4, 3, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((-10, 0)), np.asarray((0, 0))]),
            (np.asarray([[2.8, 4, 2.8, 4, 0, 0], [4, 4, 4, 4, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))])]))
    @unpack
    def testDistanceBetweenActualAndOptimalNextPosition(self, trajectoryMinDistance, trajectoryMediumDistance,
                                                        trajectoryMaxDistance):
        measurementTimeStep = 1
        getStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, self.stateIndex)
        getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(self.getSheepXPos, getStateFromTrajectory)
        distanceBetweenActualAndOptimalNextPosition = \
            DistanceBetweenActualAndOptimalNextPosition(self.computeOptimalNextPos, getPosAtNextStepFromTrajectory)

        allTrajectories = [trajectoryMinDistance, trajectoryMediumDistance, trajectoryMaxDistance]
        allDistances = [distanceBetweenActualAndOptimalNextPosition(trajectory) for trajectory in allTrajectories]

        passed = allDistances[0] < allDistances[1] < allDistances[2]
        self.assertTrue(passed)