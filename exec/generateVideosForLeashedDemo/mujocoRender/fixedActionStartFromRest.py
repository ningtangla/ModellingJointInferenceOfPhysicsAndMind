import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, TransitionFunction, IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

import mujoco_py as mujoco
import skvideo
import skvideo.io

def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', '..', 'env', 'xmls', 'chase10.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qPosInit = (-9.7, 0, -9.7, -9.7)
    qVelInit = (0, 0, 0, 0)
    numAgent = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    killzoneRadius = 0.5
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)
    numSimulationFrames = 1

    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # viewer = mujoco.MjViewer(physicsSimulation)
    frames = []

    maxRunningSteps = 50
    action = [(10, 0), (0, 0)]
    for step in range(maxRunningSteps):
        frame = physicsSimulation.render(1024, 1024, camera_name="center")
        frames.append(frame)
    skvideo.io.vwrite("testingChase10.mp4", frames)

    print("QPOS: ", physicsSimulation.data.qpos)
    print("XPOS: ", physicsSimulation.data.body_xpos)
    print("QVEL: ", physicsSimulation.data.qvel)

if __name__ == '__main__':
    main()