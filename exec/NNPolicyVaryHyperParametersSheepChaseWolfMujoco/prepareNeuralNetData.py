import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pickle

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.algorithms.mcts import CalculateScore, SelectChild, InitializeChildren, selectGreedyAction, RollOut, Expand, \
    MCTS, backup
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from exec.evaluationFunctions import GetSavePath
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy


def main():
    maxRunningSteps = 1  # 10
    qPosInit = (0, 0, 0, 0)
    numSimulations = 2  # 75

    # functions for MCTS
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 1
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionPriorUniform = {action: 1 / len(actionSpace) for action in actionSpace}
    getActionPriorUniform = lambda state: actionPriorUniform

    initializeChildren = InitializeChildren(actionSpace, sheepTransit, getActionPriorUniform)
    expand = Expand(isTerminal, initializeChildren)

    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # All agents' policies
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectGreedyAction)
    policy = lambda state: [mcts(state), stationaryAgentPolicy(state)]

    # generate trajectories
    numTrials = 2  # 1500
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco/trajectories"
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    sheepPolicyName = 'MCTS'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit, 'numSimulations': numSimulations,
                          'numTrials': numTrials, 'sheepPolicyName': sheepPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()


if __name__ == "__main__":
    main()
