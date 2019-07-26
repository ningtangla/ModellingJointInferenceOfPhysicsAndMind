import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import mujoco_py as mujoco
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.algorithms.mcts import RollOut, Expand, InitializeChildren, ScoreChild, SelectChild, MCTS, backup, \
    establishPlainActionDist
from src.neuralNetwork.policyNet import GenerateModel, restoreVariables
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle


class ApproximatePolicy:
    def __init__(self, model, actionSpace):
        self.actionSpace = actionSpace
        self.model = model

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
        graph = self.model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionIndices_ = graph.get_collection_ref("actionIndices")[0]
        actionIndices = self.model.run(actionIndices_, feed_dict={state_: stateBatch})
        actionBatch = [self.actionSpace[i] for i in actionIndices]
        if len(actionBatch) == 1:
            actionBatch = actionBatch[0]
        return actionBatch


def main():
    numSamples = 750
    maxRunningSteps = 25
    numSimulations = 100

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

    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

    # wolf NN Policy
    dirName = os.path.dirname(__file__)
    sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'trainNNEscapePolicyMujoco', 'trainedNNModels')
    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = 0.1

    NNFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                         'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                         'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    NNModelSaveExtension = ''
    getSheepModelSavePath = GetSavePath(sheepNNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    numStateSpace = 12
    numActionSpace = len(sheepActionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    initializedNNModel = generatePolicyNet(hiddenWidths)

    trainSteps = 1000
    escapeNNModelSavePath = getSheepModelSavePath({'trainSteps': trainSteps})
    restoredEscapeNNModel = restoreVariables(initializedNNModel, escapeNNModelSavePath)
    approximateSheepPolicy = ApproximatePolicy(restoredEscapeNNModel, sheepActionSpace)
    escapeNNPolicy = lambda state: {approximateSheepPolicy(state): 1}
    # transit in sheep simulation
    transitInWolfMCTS = lambda state, action: transit(state, [action, approximateSheepPolicy(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    getUniformActionPrior = lambda state: {action: 1 / numActionSpace for action in wolfActionSpace}
    initializeChildrenUniformPrior = InitializeChildren(wolfActionSpace, transitInWolfMCTS,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTS, rewardFunction, isTerminal,
                      rolloutHeuristic)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    policy = lambda state: [escapeNNPolicy(state), mcts(state)]
    trajectories = [sampleTrajectory(policy) for _ in range(numSamples)]

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'generateTrajectoriesNNSheepMCTSWolfMujoco')
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
