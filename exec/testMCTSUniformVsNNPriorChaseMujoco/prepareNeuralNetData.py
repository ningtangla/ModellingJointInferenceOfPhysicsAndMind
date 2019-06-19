import sys
sys.path.append('../../src')
sys.path.append('../../src/algorithms')
sys.path.append('../../src/sheepWolf')
sys.path.append('..')
import numpy as np
import pickle
import random

from envMujoco import Reset, IsTerminal, TransitionFunction
from mcts import CalculateScore, SelectChild, InitializeChildren, GetActionPrior, SelectNextAction, RollOut,\
HeuristicDistanceToTarget, Expand, MCTS, backup
from play import SampleTrajectory
import reward
from sheepWolfWrapperFunctions import GetAgentPosFromState
from evaluationFunctions import GetSavePath
from policiesFixed import stationaryAgentPolicy


def main():
    maxRunningSteps = 5#15
    qPosInit = (-4, 0, 4, 0)
    numSimulations = 5#200

    # functions for MCTS
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = 2
    numXPosEachAgent = 2

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex, numXPosEachAgent)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    getActionPrior = GetActionPrior(actionSpace)

    initializeChildren = InitializeChildren(actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    selectNextAction = SelectNextAction(sheepTransit)

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # All agents' policies
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)
    policy = lambda state: [mcts(state), stationaryAgentPolicy(state)]

    # generate trajectories
    numTrials = 7#100
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    sheepPolicyName = 'mcts'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit, 'numSimulations': numSimulations,
                          'numTrials': numTrials, 'sheepPolicyName': sheepPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()


if __name__ == "__main__":
    main()