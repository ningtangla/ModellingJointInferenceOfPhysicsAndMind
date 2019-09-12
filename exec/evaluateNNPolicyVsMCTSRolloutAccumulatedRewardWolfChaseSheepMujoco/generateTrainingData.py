import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from exec.evaluationFunctions import conditionDfFromParametersDict
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf

import mujoco_py as mujoco
import numpy as np
from collections import OrderedDict


class GenerateTrajectories:
    def __init__(self, numSamplesForCondition, policy, saveTrajectories, getSampleTrajectory):
        self.numSamplesForCondition = numSamplesForCondition
        self.policy = policy
        self.saveTrajectories = saveTrajectories
        self.getSampleTrajectory = getSampleTrajectory

    def __call__(self, oneConditionDf):
        maxRunningSteps = oneConditionDf.index.get_level_values('maxRunningSteps')[0]
        sampleTrajectory = self.getSampleTrajectory(maxRunningSteps)
        numSamples = self.numSamplesForCondition[maxRunningSteps]
        trajectories = [sampleTrajectory(self.policy) for _ in range(numSamples)]
        self.saveTrajectories(trajectories, oneConditionDf)


def main():
    # manipulated variables and other important parameters
    manipulatedVariables = OrderedDict()
    manipulatedVariables['maxRunningSteps'] = [10]
    numSimulations = 100
    numSamplesForCondition = {10: 1000}
    killzoneRadius = 0.5

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # Mujoco Environment
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    sheepActionInWolfSimulation = lambda state: (0, 0)
    transitInWolfMCTSSimulation = \
        lambda state, wolfSelfAction: transit(state, [sheepActionInWolfSimulation(state), wolfSelfAction])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1 / numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTSSimulation,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
                      rolloutHeuristic)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sample trajectory
    qPosInit = (0, 0, 0, 0)
    qVelInit = (0, 0, 0, 0)
    qPosInitNoise = 9.7
    qVelInitNoise = 5
    numAgent = 2
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)

    getSampleTrajectory = lambda maxRunningSteps: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset,
                                                                   chooseGreedyAction)

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco',
                                           'trainingData')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)

    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)
    saveTrajectoriesFromOneConditionDf = lambda trajectories, oneConditionDf: \
        saveAllTrajectories(trajectories, readParametersFromDf(oneConditionDf))

    # policy
    policy = lambda state: [stationaryAgentPolicy(state), mcts(state)]

    # function to generate trajectories
    generateTrajectories = GenerateTrajectories(numSamplesForCondition, policy, saveTrajectoriesFromOneConditionDf,
                                                getSampleTrajectory)

    # generate trajectories
    levelNames = list(manipulatedVariables.keys())
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)


if __name__ == '__main__':
    main()

