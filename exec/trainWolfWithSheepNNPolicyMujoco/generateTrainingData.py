import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, Reset
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy, ApproximateActionPrior, \
    ApproximateValueFunction
from src.algorithms.mcts import MCTS, SelectChild, ScoreChild, establishPlainActionDist, InitializeChildren, Expand, \
    backup
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import SaveAllTrajectories, saveToPickle, GenerateAllSampleIndexSavePaths, GetSavePath
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode


def main():
    killzoneRadius = 2
    numSimulations = 125
    maxRunningSteps = 20
    numSamples = 2000

    # mujoco env
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgentsTwoObstacles.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    numAgents = 2
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # reward function
    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    # NN Model
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # sheep policy -- NN
    sheepModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainNNEscapePolicyMujoco/trainedNNModels/killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999'
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepApproximatePolicy = ApproximatePolicy(sheepNNModel, actionSpace)
    sheepPolicy = lambda state: {sheepApproximatePolicy(state): 1}

    # wolf policy -- MCTS+NN
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    transitInWolfMCTSSimulation = lambda state, action: \
        transit(state, [sheepApproximatePolicy(state), action])

    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    wolfModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/trainedNNModels/bufferSize=2000_iteration=19999_learningRate=0.0001_maxRunningSteps=20_miniBatchSize=256_numSimulations=200_numTrajectoriesPerIteration=1'
    restoreVariables(wolfNNModel, wolfModelPath)
    approximateActionPrior = ApproximateActionPrior(wolfNNModel, actionSpace)
    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInWolfMCTSSimulation, approximateActionPrior)
    expandNNPrior = Expand(isTerminal, initializeChildrenNNPrior)

    getStateFromNode = lambda node: list(node.id.values())[0]
    approximateValue = ApproximateValueFunction(wolfNNModel)
    # estimateValueFromNode = lambda node: deathBonus if isTerminal(getStateFromNode(node)) else \
    #     approximateValue(getStateFromNode(node))
    estimateValueFromNode = EstimateValueFromNode(deathBonus, isTerminal, getStateFromNode, approximateValue)

    wolfPolicy = MCTS(numSimulations, selectChild, expandNNPrior, estimateValueFromNode, backup, establishPlainActionDist)

    # policy
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # save trajectories
    trajectoryFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'trainWolfWithSheepNNPolicyMujoco', 'trainingData')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)

    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)

    for sample in range(numSamples):
        trajectory = sampleTrajectory(policy)
        saveAllTrajectories([trajectory], {})


if __name__ == '__main__':
    main()