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
    init = [[0, 0], [-3, -3], [-5, -5]] + [[-3-0.2*(i), -3-0.2*(i)] for i in range(1, 10)]
    physicsSimulation = mujoco.MjSim(physicsModel)
    #physicsSimulation.model.body_pos[-12: , :2] = init
    __import__('ipdb').set_trace()
    physicsSimulation.model.body_mass[8] = 10000
    #physicsSimulation.data.body_xpos[-12: , :2] = init
    #physicsSimulation.data.qpos[:] = np.array(init).flatten()
    physicsSimulation.set_constants()
    print(physicsSimulation.model.body_pos)
    print(physicsSimulation.data.body_xpos)
    print(physicsSimulation.data.qpos)
    print(physicsSimulation.data.qvel)
    physicsSimulation.forward()
    print(physicsSimulation.model.body_pos)
    print(physicsSimulation.data.body_xpos)
    print(physicsSimulation.data.qpos)
    print(physicsSimulation.data.qvel)
    
    baselinePhysicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'chase10.xml')
    baselinePhysicsModel = mujoco.load_model_from_path(baselinePhysicsDynamicsPath)
    #baselinePhysicsModel.body_pos[-12: , :2] = init
    baselinePhysicsSimulation = mujoco.MjSim(baselinePhysicsModel)
    #baselinePhysicsSimulation.set_constants()
    print(baselinePhysicsSimulation.model.body_pos)
    print(baselinePhysicsSimulation.data.body_xpos)
    print(baselinePhysicsSimulation.data.qpos)
    print(baselinePhysicsSimulation.data.qvel)
    baselinePhysicsSimulation.forward()
    print(baselinePhysicsSimulation.model.body_pos)
    print(baselinePhysicsSimulation.data.body_xpos)
    print(baselinePhysicsSimulation.data.qpos)
    print(baselinePhysicsSimulation.data.qvel)
    #__import__('ipdb').set_trace()
    
    
    physicsViewer = mujoco.MjViewer(physicsSimulation)
    numSimulationFrames = 10000
    for frameIndex in range(numSimulationFrames):
        if frameIndex > 500:
            action = np.array([1, 1, 1, 1, 0, 0])
            physicsSimulation.data.ctrl[:] = action
        physicsSimulation.step()
        physicsSimulation.forward()
        physicsViewer.render()
    
    #print(physicsSimulation.model.body_pos)
    #print(physicsSimulation.data.body_xpos)
    #print(physicsSimulation.data.qpos)
    
    baselinePhysicsViewer = mujoco.MjViewer(baselinePhysicsSimulation)
    numSimulationFrames = 1000
    for frameIndex in range(numSimulationFrames):
        #action = np.array([0] * 24)
        #baselinePhysicsSimulation.data.ctrl[:] = action
        baselinePhysicsSimulation.step()
        baselinePhysicsSimulation.forward()
        baselinePhysicsViewer.render()
    #print(baselinePhysicsSimulation.model.body_pos)
    #print(baselinePhysicsSimulation.data.body_xpos)
    #print(baselinePhysicsSimulation.data.qpos)
    
if __name__ == '__main__':
    main()
