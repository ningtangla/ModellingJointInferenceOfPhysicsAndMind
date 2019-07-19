import sys
import os
import mujoco_py as mujoco
import numpy as np

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))


from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


def main():
    # transition function
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'leasedChasing.xml')
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'chase10.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    initQPos = np.zeros(46)
    physicsSimulation.data.qpos[:] = initQPos
    action = np.array([0,0,-1,-1,0,0])
    physicsSimulation.data.ctrl[:] = action
    physicsViewer = mujoco.MjViewer(physicsSimulation)
    numSimulationFrames = 100000
    for frameIndex in range(numSimulationFrames):
        physicsSimulation.step()
        physicsSimulation.forward()
        physicsViewer.render()

if __name__ == '__main__':
    main()
