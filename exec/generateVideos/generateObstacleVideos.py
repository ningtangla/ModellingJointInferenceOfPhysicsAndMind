import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import TransitionFunction, Reset, IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.trajectoriesSaveLoad import saveToPickle
from src.episode import SampleTrajectory, chooseGreedyAction


def main():
    # env mujoco
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgentsTwoObstacles.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    numAgents = 2
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # NN Model
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # sheep policy
    sheepModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainNNEscapePolicyMujoco/trainedNNModels/killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999'
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepApproximatePolicy = ApproximatePolicy(sheepNNModel, actionSpace)
    sheepPolicy = lambda state: {sheepApproximatePolicy(state): 1}

    # wolf policy
    wolfModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/trainedNNModels/bufferSize=2000_iteration=19999_learningRate=0.0001_maxRunningSteps=20_miniBatchSize=256_numSimulations=200_numTrajectoriesPerIteration=1'
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, wolfModelPath)
    wolfApproximatePolicy = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = lambda state: {wolfApproximatePolicy(state): 1}

    # policy
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    # sample trajectory
    maxRunningSteps = 50
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    numSamples = 5
    trajectories = [sampleTrajectory(policy) for _ in range(numSamples)]

    # save trajectory
    [saveToPickle(trajectory, 'sampleIndex{}.pickle'.format(sampleIndex)) for trajectory, sampleIndex in zip(trajectories, range(numSamples))]


if __name__ == '__main__':
    main()