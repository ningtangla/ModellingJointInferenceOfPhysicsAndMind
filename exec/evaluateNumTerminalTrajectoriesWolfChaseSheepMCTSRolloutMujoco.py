import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from collections import OrderedDict
import mujoco_py as mujoco
import numpy as np
import time
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.algorithms.mcts import RollOut, Expand, InitializeChildren, ScoreChild, SelectChild, MCTS, backup, \
    establishPlainActionDist
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from exec.evaluationFunctions import GenerateInitQPosUniform, ComputeStatistics, conditionDfFromParametersDict
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy


class GenerateTrajectories:
    def __init__(self, getMCTS, sheepPolicy, allSampleTrajectories, saveTrajectories):
        self.getMCTS = getMCTS
        self.sheepPolicy = sheepPolicy
        self.allSampleTrajectories = allSampleTrajectories
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        startTime = time.time()
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        mcts = self.getMCTS(numSimulations)
        policy = lambda state: [self.sheepPolicy(state), mcts(state)]
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in self.allSampleTrajectories]
        self.saveTrajectories(trajectories, oneConditionDf)
        endTime = time.time()
        print("time taken for numSim = {} is {} seconds".format(numSimulations, (endTime-startTime)))


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [50, 75, 100, 200, 300]
    numSamples = 100
    maxRunningSteps = 20

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
    killzoneRadius = 2
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

    getMCTS = lambda numSimulations: MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sampleTrajectory
    numAgents = 2
    qVelInitNoise = 0
    qPosInitNoise = 0
    getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgents,
                                                              qPosInitNoise, qVelInitNoise)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
    allQPosInit = [generateQPosInit() for _ in range(numSamples)]
    qVelInitRange = 10
    allQVelInit = np.random.uniform(-qVelInitRange, qVelInitRange, (numSamples, 4))
    getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, allQPosInit[trial], allQVelInit[trial], numAgents,
                                                   qPosInitNoise, qVelInitNoise)
    getSampleTrajectory = lambda trial: SampleTrajectory(maxRunningSteps, transit, isTerminal, getResetFromTrial(trial),
                                                         chooseGreedyAction)
    allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(numSamples)]

    # save trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', 'data',
                                           'evaluateNumTerminalTrajectoriesWolfChaseSheepMCTSRolloutMujoco')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveParameters = {'killzoneRadius': killzoneRadius, 'qVelInitRange': qVelInitRange,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)
    saveTrajectoriesFromDf = lambda trajectories, df: saveAllTrajectories(trajectories, readParametersFromDf(df))

    # generate trajectories
    generateTrajectories = GenerateTrajectories(getMCTS, stationaryAgentPolicy, allSampleTrajectories, saveTrajectoriesFromDf)
    # toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # computeStatistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: 1 if trajectory[-1][1] is None else 0
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    statisticsDf.plot(y='mean', marker='o')
    plt.ylabel('Fraction of trajectories where the prey is caught')
    plt.title("Killzone radius = {}, rollout with heuristic, initial velocities sampled from -{} to {}".format(
        killzoneRadius, qVelInitRange, qVelInitRange))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()