import sys
import os
import mujoco_py as mujoco
import copy

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack
from mujoco_py import load_model_from_path, MjSim

# Local import
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


@ddt
class TestEnvMujoco(unittest.TestCase):
    def setUp(self):
        pass

    @data((np.asarray([[1, 2, 1, 2, 0, 0], [4, 5, 4, 5, 0, 0]]), np.asarray([[1, 1], [1, 1]]), 5, 10),
          (np.asarray([[9, 9, 9, 9, 0, 0], [-9, -9, -9, -9, 0, 0]]), np.asarray([[1, 1], [1, 1]]), 5, 10),
          (np.asarray([[1, 2, 1, 2, 0, 0], [4, 5, 4, 5, 0, 0]]), np.asarray([[1, 1], [1, 1]]), 8, 12))
    @unpack
    def testMassEffectInTransition(self, state, allAgentsActions, smallMass, largeMass):
        # transition function
        dirName = os.path.dirname(__file__)
        physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
        agentsBodyMassIndex = [6, 7]
        physicsSmallMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSmallMassModel.body_mass[agentsBodyMassIndex] = [smallMass, smallMass]
        physicsLargeMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsLargeMassModel.body_mass[agentsBodyMassIndex] = [largeMass, largeMass]
        physicsSmallMassSimulation = mujoco.MjSim(physicsSmallMassModel)
        physicsLargeMassSimulation = mujoco.MjSim(physicsLargeMassModel)
        # set_constants fit for mujoco_py version >= 2.0, no fit for 1.50
        physicsSmallMassSimulation.set_constants()
        physicsLargeMassSimulation.set_constants()

        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
        killzoneRadius = 2
        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transitSmallMassAgents = TransitionFunction(physicsSmallMassSimulation, isTerminal, numSimulationFrames)
        transitLargeMassAgents = TransitionFunction(physicsLargeMassSimulation, isTerminal, numSimulationFrames)
        nextStateInSmallMassTransition = transitSmallMassAgents(state, allAgentsActions)
        nextStateInLargeMassTransition = transitLargeMassAgents(state, allAgentsActions)
        stateChangeInSmallMassTransition = nextStateInSmallMassTransition - state
        stateChangeInLargeMassTransition = nextStateInLargeMassTransition - state

        trueMassRatio = smallMass / largeMass
        self.assertTrue(np.allclose(stateChangeInLargeMassTransition, stateChangeInSmallMassTransition * trueMassRatio))


if __name__ == '__main__':
    unittest.main()
