import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle

import mujoco_py as mujoco


def main():
    numSamples = 10000
    maxRunningSteps = 20
    policyName = 'NNIterative20kTrainSteps'

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # restore variables
    wolfNNModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/trainedNNModels/bufferSize=2000_iteration=19999_learningRate=0.0001_maxRunningSteps=20_miniBatchSize=256_numSimulations=200_numTrajectoriesPerIteration=1'
    restoreVariables(wolfNNModel, wolfNNModelPath)
    wolfActionForState = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = lambda state: {wolfActionForState(state): 1}

    # policy
    policy = lambda state: [stationaryAgentPolicy(state), wolfPolicy(state)]

    # mujoco env
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 8
    qPosInitNoise = 9.7

    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate all trajectories
    trajectories = [sampleTrajectory(policy) for _ in range(numSamples)]

    # save trajectories
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'policyName': policyName}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainMCTSNNIteratively',
                                           'replayBufferStartWithTrainedModel', 'evaluateValueNetChasing',
                                           'evaluationTrajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # save all trajectories in a single pickle
    trajectoriesSavePath = getTrajectorySavePath({})
    saveToPickle(trajectories, trajectoriesSavePath)


if __name__ == '__main__':
    main()