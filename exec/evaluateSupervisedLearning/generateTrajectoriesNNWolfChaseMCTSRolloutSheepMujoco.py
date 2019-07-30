import time
import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import mujoco_py as mujoco
import numpy as np
import glob

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.algorithms.mcts import RollOut, Expand, InitializeChildren, ScoreChild, SelectChild, MCTS, backup, \
    establishPlainActionDist
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


class GenerateAllSampleIndexSavePaths:
    def __init__(self, getSavePath):
        self.getSavePath = getSavePath

    def __call__(self, numSamples, pathParameters):
        parametersWithSampleIndex = lambda sampleIndex: dict(list(pathParameters.items()) + [('sampleIndex', sampleIndex)])
        genericSavePath = self.getSavePath(parametersWithSampleIndex('*'))
        existingFilesNames = glob.glob(genericSavePath)
        numExistingFiles = len(existingFilesNames)
        allIndexParameters = {sampleIndex: parametersWithSampleIndex(sampleIndex+numExistingFiles) for sampleIndex in
                              range(numSamples)}
        allSavePaths = {sampleIndex: self.getSavePath(indexParameters) for sampleIndex, indexParameters in
                        allIndexParameters.items()}

        return allSavePaths


class SaveAllTrajectories:
    def __init__(self, saveData, generateAllSampleIndexSavePaths):
        self.saveData = saveData
        self.generateAllSampleIndexSavePaths = generateAllSampleIndexSavePaths

    def __call__(self, trajectories, pathParameters):
        numSamples = len(trajectories)
        allSavePaths = self.generateAllSampleIndexSavePaths(numSamples, pathParameters)
        saveTrajectory = lambda sampleIndex: self.saveData(trajectories[sampleIndex], allSavePaths[sampleIndex])
        [saveTrajectory(sampleIndex) for sampleIndex in range(numSamples)]

        return None


def main():
    numSamples = 500
    maxRunningSteps = 25
    numSimulations = 100
    wolfNNModelPath = '../../data/repeatTrainNNEscapePolicyMujoco/wolfNNModels/killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    qPosInit = (0, 0, 0, 0)
    qVelInit = (0, 0, 0, 0)
    numAgents = 2
    qVelInitNoise = 8
    qPosInitNoise = 9.7
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # wolf NN Policy
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoredNNModel = restoreVariables(initializedNNModel, wolfNNModelPath)
    approximatePolicy = ApproximatePolicy(restoredNNModel, actionSpace)
    NNPolicy = lambda state: {approximatePolicy(state): 1}

    # transit in sheep simulation
    transitInSheepMCTS = lambda state, action: transit(state, [action, approximatePolicy(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTS,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = -0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTS, rewardFunction, isTerminal,
                      rolloutHeuristic)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    policy = lambda state: [mcts(state), NNPolicy(state)]
    # trajectories = [sampleTrajectory(policy) for _ in range(numSamples)]
    trajectories = []
    for i in range(numSamples):
        startT = time.time()
        traj = sampleTrajectory(policy)
        trajectories.append(traj)
        endT = time.time()
        print(f"traj #{i+1} took {endT - startT:.1f}s")

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'repeatTrainNNEscapePolicyMujoco', 'trainingTrajectoriesNew')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)

    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)

    saveAllTrajectories(trajectories, {})


if __name__ == '__main__':
    main()
