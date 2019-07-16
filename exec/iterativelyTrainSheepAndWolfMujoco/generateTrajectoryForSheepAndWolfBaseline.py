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
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def main():
    numSamples = 1
    maxRunningSteps = 25
    numSimulations = 10

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

    # wolf heatSeeking Policy
    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getSheepXPos, getWolfXPos,
                                                        computeAngleBetweenVectors)

    # transit in sheep simulation
    transitInSheepMCTS = lambda state, action: transit(state, [action, chooseGreedyAction(wolfPolicy(state))])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1 / numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTS,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTS, rewardFunction, isTerminal,
                      rolloutHeuristic)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    policy = lambda state: [mcts(state), wolfPolicy(state)]
    trajectories = [sampleTrajectory(policy) for _ in range(numSamples)]

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'generateTrajectoryForSheepAndWolfBaseline')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)

    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)

    path = getTrajectorySavePath(trajectorySaveParameters)
    saveToPickle(trajectories, path)


if __name__ == '__main__':
    main()
