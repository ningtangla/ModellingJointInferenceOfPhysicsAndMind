import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, TransitionFunction, IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy

import mujoco_py as mujoco
import skvideo
import skvideo.io

def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qPosInit = (0, 0, -9.7, -9.7)
    qVelInit = (0, 1, 0, 0)
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
    numSimulationFrames = 20

    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    heatSeekingPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfXPos, getSheepXPos, 5)
    heatSeekingAction = lambda state: list(heatSeekingPolicy(state).keys())[0]

    frames = []

    maxRunningSteps = 50
    state = reset()
    action = lambda state: [(0, 0), heatSeekingAction(state)]
    for step in range(maxRunningSteps):
        newState = transit(state, action(state))
        state = newState
        if step == 25:
            print(state)
    #     frame = physicsSimulation.render(1024, 1024, camera_name="center")
    #     frames.append(frame)
    # skvideo.io.vwrite("heatSeekingWolfConstantVelocitySheep.mp4", frames)


if __name__ == '__main__':
    main()