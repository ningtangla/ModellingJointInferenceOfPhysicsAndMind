import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.envMujoco import Reset
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import saveToPickle

def main():
    # transition function
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # reset function
    qPosInit = (0, 0, 0, 0)     # (initial position of sheep, initial position of wolf)
    qVelInit = (0, 0, 0, 0)     # (initial velocity of sheep, initial velocity of wolf)
    qPosInitNoise = 9.7         # adds some randomness to the initial positions
    qVelInitNoise = 5           # adds some randomness to the initial velocities
    numAgent = 2
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)

    # sample trajectory
    maxRunningSteps = 10        # max possible length of the trajectory/episode
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
    approximateWolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace)

    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..','NNModels','sheepNNModels', 'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    approximateSheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace)

    approximatePolicyList = [approximateSheepPolicy, approximateWolfPolicy]
    policy = lambda state: [{approximatePolicy(state): 1} for approximatePolicy in approximatePolicyList]

    trajectory = sampleTrajectory(policy)
    dataIndex = 5
    dataPath = os.path.join(dirName, '..', 'trainedData', 'trajectory'+ str(dataIndex) + '.pickle')
    saveToPickle(trajectory, dataPath)


if __name__ == '__main__':
    main()
