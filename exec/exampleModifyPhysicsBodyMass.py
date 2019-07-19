import sys
import os
import mujoco_py as mujoco

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))


from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


def main():
    # transition function
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    agentsBodyMassIndex = [6, 7]
    physicsSmallMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSmallMassModel.body_mass[agentsBodyMassIndex] = [4, 5] 
    physicsLargeMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsLargeMassModel.body_mass[agentsBodyMassIndex] = [8, 10] 
    physicsSmallMassSimulation = mujoco.MjSim(physicsSmallMassModel)
    physicsLargeMassSimulation = mujoco.MjSim(physicsLargeMassModel)
    #set_constants fit for mujoco_py version >= 2.0, no fit for 1.50 
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

if __name__ == '__main__':
    main()
