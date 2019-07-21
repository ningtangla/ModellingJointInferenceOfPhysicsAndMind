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
        self.agentJointIndex = list(range(self.nAgent * self.nDimJointEachBody))
        self.ropeJointIndex = list(range(self.nAgent * self.nDimJointEachBody, self.nJoint))
        self.nRopeJoint = len(self.ropeJointIndex)
        self.nRopeBody = int(self.nRopeJoint / self.nDimJointEachBody)
    
    def testXMLState(self):
        qPosFromXML = self.simulation.data.qpos
        qVelFromXML = self.simulation.data.qvel
        self.assertTrue(np.all(qPosFromXML == 0))
        self.assertTrue(np.all(qVelFromXML == 0))
   
    @data((np.array([-3, 3, 5, 5, -8, -8]), np.array([0, 0, 1, 1, -1, -1])))
    @unpack
    def testSetState(self, agentQPosToSet, action):
        qPos = np.concatenate([agentQPosToSet, np.zeros(self.nRopeJoint)])
        qPosFromXML = self.simulation.data.qpos
        self.assertTrue(not np.all(qPos == qPosFromXML))
        
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.ctrl[:] = action
        self.simulation.step()
        self.simulation.forward()
        newQPos = self.simulation.data.qpos[:]
        self.assertTrue(not np.all(qPos == newQPos))
        
        self.simulation.data.qpos[:] = qPos
        self.assertTrue(np.all(qPos == self.simulation.data.qpos[:]))
        
        physicsViewer = mujoco.MjViewer(self.simulation)
        numSimulationFrames = 100
        for frameIndex in range(numSimulationFrames):
            self.simulation.data.qpos[:] = qPos
            self.simulation.step()
            self.simulation.forward()
            physicsViewer.render()

    @data((np.array([-3, 3, 5, 5, -8, -8]), np.array([0, 0, 1, 1, -1, -1])))
    @unpack
    def testXPosEqualsQPos(self, agentQPosToSet, action):
        qPos = np.concatenate([agentQPosToSet, np.zeros(self.nRopeJoint)])
        self.simulation.data.qpos[:] = qPos
        self.simulation.forward()
        xPos2Dim = np.array([siteXPos[0:2] for siteXPos in self.simulation.data.site_xpos]).flatten()
        self.assertTrue(np.all(qPos == xPos2Dim))


if __name__ == '__main__':
    unittest.main()
