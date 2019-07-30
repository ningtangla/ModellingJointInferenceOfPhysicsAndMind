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
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'leased.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    initQPos = np.array([1,1,2,2,-4,-4] + [0]*18)
    physicsSimulation.data.qpos[:] = initQPos
    physicsSimulation.step()
    physicsSimulation.forward()
    action = np.array([0,0,-1,-1,0,0])
    #physicsSimulation.data.ctrl[:] = action
    physicsViewer = mujoco.MjViewer(physicsSimulation)
    numSimulationFrames = 1000
    initQPos = np.array([9,-9, 4, 4, -4,-4] + [0]*18)
    physicsSimulation.data.qpos[:] = initQPos
    for frameIndex in range(numSimulationFrames):
        physicsSimulation.step()
        physicsSimulation.forward()
        physicsViewer.render()

if __name__ == '__main__':
    main()
