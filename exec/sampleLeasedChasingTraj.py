import sys
import os
import mujoco_py as mujoco
import numpy as np

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import saveToPickle
from src.inferChasing.continuousPolicy import RandomPolicy


def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'leased.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    # initQPos = np.array([1,1,2,2,-4,-4] + [0]*18)
    # physicsSimulation.data.qpos[:] = initQPos
    # physicsSimulation.step()
    # physicsSimulation.forward()

    # action = np.array([0,0,-1,-1,0,0])
    # physicsSimulation.data.ctrl[:] = action
    # physicsViewer = mujoco.MjViewer(physicsSimulation)
    numSimulationFrames = 20
    # initQPos = np.array([9,-9, 4, 4, -4,-4] + [0]*18)

    # physicsSimulation.data.qpos[:] = initQPos
    # for frameIndex in range(numSimulationFrames):
    #     physicsSimulation.step()
    #     physicsSimulation.forward()
    #     # physicsViewer.render()

    sheepId = 0
    wolfId = 1
    qPosIndex=[0,1]
    getSheepXPos = GetAgentPosFromState(sheepId, qPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, qPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    initQVel = (0,) * 24
    initQPos = np.array([1,1, 0, 0, -1,-1] + [0]*18)

    qPosInitNoise = 7
    qVelInitNoise = 5
    numAgent = 3
    tiedAgentId = [1, 2]
    ropeParaIndex = list(range(3, 12))
    maxRopePartLength = 0.25
    reset = ResetUniform(physicsSimulation, initQPos, initQVel, numAgent)
    print('reset', reset())
    # sample trajectory
    maxRunningSteps = 20        # max possible length of the trajectory/episode
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)


    # Neural Network
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # wolf NN Policy
    wolfModelPath = os.path.join(dirName, '..','NNModels','wolfNNModels', 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, wolfModelPath)
    wolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace) # input state, return action distribution

    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..','NNModels','sheepNNModels', 'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace) # input sheepstate, return action distribution

    randomPolicy = RandomPolicy(actionSpace)
    getWolfSheepState = lambda state: [[state[0][:6]], [state[1][:6]]]
    policy = lambda state: [sheepPolicy(getWolfSheepState(state)), wolfPolicy(getWolfSheepState(state)), randomPolicy(state)]

    trajectory = sampleTrajectory(policy)
    dataIndex = 5
    dataPath = os.path.join(dirName, '..', 'trainedData', 'NNleasedTraj'+ str(dataIndex) + '.pickle')
    saveToPickle(trajectory, dataPath)


if __name__ == '__main__':
    main()
