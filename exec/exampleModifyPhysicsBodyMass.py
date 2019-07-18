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
    sheepBodyMassIndex = 6
    wolfBodyMassIndex = 7
    smallMass = 5
    largeMass = 10
    physicsSmallMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSmallMassModel.body_mass[[sheepBodyMassIndex, wolfBodyMassIndex]] = [smallMass, smallMass] 
    physicsLargeMassModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsLargeMassModel.body_mass[[sheepBodyMassIndex, wolfBodyMassIndex]] = [largeMass, largeMass] 
    physicsSmallMassSimulation = mujoco.MjSim(physicsSmallMassModel)
    physicsLargeMassSimulation = mujoco.MjSim(physicsLargeMassModel)

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
