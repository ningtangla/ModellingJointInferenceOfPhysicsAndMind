
import sys
import os
import mujoco_py as mujoco
import numpy as np

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))


class TestEnvMujoco(unittest.TestCase):
    def setUp():
        self.modelPath = os.path.join(DIRNAME, '..', 'env', 'xmls', 'leased.xml')
        self.model = load_model_from_path(self.modelPath)
        self.simulation = MjSim(self.model)
        self.nDimQPos = len(self.simulation.data.qPos)
        self.nDimQVel = len(self.simulation.data.qVel)
        self.nDimJointEachBody = 2
        self.nJoint = self.simulation.model.njnt
        self.nAgent = 3
        self.agentJointIndex = list(range(self.nAgent))
        self.ropeJointIndex = list(range(self.nJoint/self.nDimJointEachBody)) - self.agentJointIndex
        self.nRopeBody = len(self.ropeJointIndex) / self.nDimJoint
    
    @data((np.array([-3, 3, 5, 5, -8, -8]+[0]*int(self.nRopeBody)), np.array([0, 0, 1, 1, -1, -1]))
    @unpack
    def testSetState(self, qPosToSet, action)
        qPosFromXML = self.physicsSimulation.data.qPos
        qVelFromXML = self.physicsSimulation.data.
        self.assertTrue(np.all(
        physicsSimulation.data.ctrl[:] = action
        physicsViewer = mujoco.MjViewer(physicsSimulation)
        physicsSimulation.step()
        newQPos = 
        numSimulationFrames = 100000
        for frameIndex in range(numSimulationFrames):
            physicsSimulation.step()
            physicsSimulation.forward()
            physicsViewer.render()

if __name__ == '__main__':
    main()
