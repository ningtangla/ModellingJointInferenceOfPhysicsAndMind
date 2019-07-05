import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, unpack, data
import numpy as np
from mujoco_py import load_model_from_path, MjSim

from src.constrainedChasingEscapingEnv.measure import ComputeOptimalNextPos, DistanceBetweenActualAndOptimalNextPosition, \
    calculateCrossEntropy
from src.constrainedChasingEscapingEnv.wrappers import GetStateFromTrajectory, GetAgentPosFromState, GetAgentPosFromTrajectory
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.envMujoco import TransitionFunction, IsTerminal
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.play import agentDistToGreedyAction

@ddt
class TestMeasurementFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(self.actionSpace, self.getSheepXPos,
                                                                    self.getWolfXPos, computeAngleBetweenVectors)
        self.getOptimalAction = lambda state: agentDistToGreedyAction(self.optimalPolicy(state))
        self.killzoneRadius = 0.5
        self.isTerminal = IsTerminal(self.killzoneRadius, self.getSheepXPos, self.getWolfXPos)
        self.dirName = os.path.dirname(__file__)
        self.mujocoModelPath = os.path.join(self.dirName, '..', 'env', 'xmls', 'twoAgents.xml')
        self.mujocoModel = load_model_from_path(self.mujocoModelPath)
        self.simulation = MjSim(self.mujocoModel)
        self.numSimulationFrames = 20
        self.transit = TransitionFunction(self.simulation, self.isTerminal, self.numSimulationFrames)

        self.stationaryAgentAction = lambda state: agentDistToGreedyAction(stationaryAgentPolicy(state))
        self.sheepTransit = lambda state, action: self.transit(state, [action, self.stationaryAgentAction(state)])
        self.stateIndex = 0
        self.getInitStateFromTrajectory = GetStateFromTrajectory(0, self.stateIndex)
        self.computeOptimalNextPos = ComputeOptimalNextPos(self.getInitStateFromTrajectory, self.getOptimalAction,
                                                           self.sheepTransit, self.getSheepXPos)

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

    @data(({"predict":np.array([0.228, 0.619, 0.153]), "target":np.array([0, 1, 0])}, 0.47965),
        ({"predict":np.array([0, 1, 0]), "target":np.array([0, 1, 0])}, 0))
    @unpack
    def testCrossEntropy(self, data, groundTruth):
        self.assertAlmostEqual(calculateCrossEntropy(data), groundTruth, places=5)

if __name__ == "__main__":
    unittest.main()
