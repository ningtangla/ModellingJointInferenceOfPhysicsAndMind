import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import mujoco_py as mujoco
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt


from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.algorithms.mcts import RollOut, Expand, InitializeChildren, ScoreChild, SelectChild, MCTS, backup, \
    establishPlainActionDist
from src.episode import SampleTrajectoryTerminationProbability, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from exec.evaluationFunctions import GenerateInitQPosUniform, ComputeStatistics, conditionDfFromParametersDict
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy


class GenerateTrajectories:
    def __init__(self, numSamples, policy, getSampleTrajectory, saveTrajectories):
        self.numSamples = numSamples
        self.policy = policy
        self.getSampleTrajectory = getSampleTrajectory
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        terminationProb = oneConditionDf.index.get_level_values('terminationProbability')[0]
        print('termination probability: ', terminationProb)
        sampleTrajectory = self.getSampleTrajectory(terminationProb)
        trajectories = [sampleTrajectory(self.policy) for _ in range(self.numSamples)]
        self.saveTrajectories(trajectories, oneConditionDf)

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['terminationProbability'] = [1/80, 1/40, 1/30, 1/20, 1/10, 1/2]
    numSimulations = 75
    numSamples = 500

    levelNames = list(manipulatedVariables.keys())
    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    sheepActionInWolfMCTS = lambda state: np.array([0, 0])
    transitInWolfMCTS = lambda state, action: transit(state, [sheepActionInWolfMCTS(state), action])

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTS,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTS, rewardFunction, isTerminal,
                      rolloutHeuristic)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sampleTrajectory
    qVelInit = (0, 0, 0, 0)
    qPosInit = (8, 8, -8, -8)
    numAgents = 2
    qVelInitNoise = 0
    qPosInitNoise = 0
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    getSampleTrajectory = lambda terminationProbability: \
        SampleTrajectoryTerminationProbability(terminationProbability, transit, isTerminal, reset, chooseGreedyAction)

    # save trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', 'data',
                                           'evaluateMeanEpisodeLengthVaryTerminationProbability', 'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)
    saveTrajectoriesFromDf = lambda trajectories, df: saveAllTrajectories(trajectories, readParametersFromDf(df))

    # generate trajectories
    policy = lambda state: [stationaryAgentPolicy(state), stationaryAgentPolicy(state)]
    generateTrajectories = GenerateTrajectories(numSamples, policy, getSampleTrajectory, saveTrajectoriesFromDf)
    # toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print(statisticsDf)

    # plot
    statisticsDf.plot(y='mean', marker='o', yerr='std')
    plt.ylabel('Mean Episode Length')
    plt.show()


if __name__ == '__main__':
    main()

