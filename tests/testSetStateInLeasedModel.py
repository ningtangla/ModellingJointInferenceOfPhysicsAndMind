import unittest
from ddt import ddt, unpack, data
import sys
import os
import mujoco_py as mujoco
import numpy as np

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

@ddt
class TestEnvMujoco(unittest.TestCase):
    def setUp(self):
        self.modelPath = os.path.join(DIRNAME, '..', 'env', 'xmls', 'leased.xml')
        self.model = mujoco.load_model_from_path(self.modelPath)
        self.simulation = mujoco.MjSim(self.model)
        self.nDimQPos = len(self.simulation.data.qpos)
        self.nDimQVel = len(self.simulation.data.qvel)
        self.nDimJointEachBody = 2
        self.nJoint = self.simulation.model.njnt
        self.nAgent = 3
        self.agentJointIndex = list(range(self.nAgent))
        self.ropeJointIndex = list(range(self.nAgent, int(self.nJoint/self.nDimJointEachBody)))
        self.nRopeBody = int(len(self.ropeJointIndex) / self.nDimJointEachBody)
    
    def testXMLState(self):
        qPosFromXML = self.simulation.data.qpos
        qVelFromXML = self.simulation.data.qvel
        self.assertTrue(np.all(qPosFromXML == 0))
        self.assertTrue(np.all(qVelFromXML == 0))
   
    @data((np.array([-3, 3, 5, 5, -8, -8]), np.array([0, 0, 1, 1, -1, -1])))
    @unpack
    def testSetState(self, agentQPosToSet, action):
        
        physicsSimulation.data.ctrl[:] = action
        physicsViewer = mujoco.MjViewer(physicsSimulation)
        physicsSimulation.step()
        numSimulationFrames = 100000
        for frameIndex in range(numSimulationFrames):
            physicsSimulation.step()
            physicsSimulation.forward()
            physicsViewer.render()

if __name__ == '__main__':
    unittest.main()
