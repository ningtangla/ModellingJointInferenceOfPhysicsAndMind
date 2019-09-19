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
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'noRopeCollision3AgentsWithFriction.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    init = [[0, -0, 0], [-7, -7, 0], [-9, -9, 0]] + [[-7-0.2*(i), -7-0.2*(i), 0] for i in range(1, 10)]
    physicsSimulation = mujoco.MjSim(physicsModel)
    #physicsSimulation.model.body_pos[-12: , :2] = init
    physicsSimulation.model.body_mass[8] = 13
    physicsSimulation.model.geom_friction[:,0] = 1
    #physicsSimulation.model.tendon_range[:] = [[0, 0.7]]*10
    #physicsSimulation.data.body_xpos[-12: , :2] = init
    physicsSimulation.set_constants()
    physicsSimulation.forward()
    print(physicsSimulation.model.body_pos)
    print(physicsSimulation.data.body_xpos)
    print(physicsSimulation.data.qpos)
    print(physicsSimulation.data.qvel)
    physicsSimulation.data.qpos[:] = np.array(init).flatten()
    #physicsSimulation.step()
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
    numSimulationFrames = 2000
    totalMaxVel = 0
    print(physicsSimulation.data.qvel, '!!!')
    print(physicsSimulation.data.qpos, '~~~')
    print(physicsSimulation.data.body_xpos, '...')
    for frameIndex in range(numSimulationFrames):
        if frameIndex == 850 or frameIndex == 900:
            print(physicsSimulation.data.ctrl[:], '###')
            print(physicsSimulation.data.qvel, '!!!')
            print(physicsSimulation.data.qpos, '~~~')
            print(physicsSimulation.data.body_xpos, '...')
        if frameIndex % 20 == 0 and frameIndex > 300:
            action = np.array([-100, 100, 0, 13, 13, 0, -4, -4, 0])
            physicsSimulation.data.ctrl[:] = action
        vels = physicsSimulation.data.qvel
        maxVelInAllAgents = vels[2]
        #maxVelInAllAgents = max([np.linalg.norm(vel) for vel in vels])
        if maxVelInAllAgents > totalMaxVel:
            totalMaxVel = maxVelInAllAgents
        physicsSimulation.step()
        physicsSimulation.forward()
        physicsViewer.render()

    print(totalMaxVel)
    #print(physicsSimulation.model.body_pos)
    #print(physicsSimulation.data.body_xpos)
    #print(physicsSimulation.data.qpos)
    
    baselinePhysicsViewer = mujoco.MjViewer(baselinePhysicsSimulation)
    numSimulationFrames = 0
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
