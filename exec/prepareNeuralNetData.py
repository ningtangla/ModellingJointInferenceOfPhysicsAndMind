import sys
sys.path.append('../src')
sys.path.append('../src/algorithms')
sys.path.append('../src/sheepWolf')
import numpy as np
import pickle
import random

from envMujoco import Reset, IsTerminal, TransitionFunction
from mcts import CalculateScore, SelectChild, InitializeChildren, GetActionPrior, SelectNextAction, RollOut,\
HeuristicDistanceToTarget, Expand, MCTS, backup
from play import SampleTrajectory
import reward
from wrapperFunctions import GetAgentPos
from evaluationFunctions import GetSavePath
from policiesFixed import stationaryAgentPolicy

class SampleTrajectorySingleAgentActions:
    def __init__(self, sampleTrajectory, agentId):
        self.sampleTrajectory = sampleTrajectory
        self.agentId = agentId

    def __call__(self, policy):
        trajectory = self.sampleTrajectory(policy)
        trajectorySingleAgentActions = [(state, action[self.agentId]) if action is not None else (state, None)
                                    for state, action in trajectory]

        return trajectorySingleAgentActions


def generateData(sampleTrajectory, policy, actionSpace, trajNumber, path):
    totalStateBatch = []
    totalActionBatch = []
    for index in range(trajNumber):
        if index % 100 == 0: print(index)
        trajectory = sampleTrajectory(policy)
        states, actions = zip(*trajectory)
        totalStateBatch = totalStateBatch + list(states)
        oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in
                          range(len(actionSpace))] for action in actions]
        totalActionBatch = totalActionBatch + oneHotActions

    dataSet = list(zip(totalStateBatch, totalActionBatch))
    saveFile = open(path, "wb")
    pickle.dump(dataSet, saveFile)


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()
    return dataSet


def sampleData(data, batchSize):
    batch = random.sample(data, batchSize)
    batchInput = [x for x, _ in batch]
    batchOutput = [y for _, y in batch]
    return batchInput, batchOutput


def main():
    maxRunningSteps = 15
    qPosInit = (-4, 0, 4, 0)
    numSimulations = 200

    # reset function
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # functions to get agent positions
    sheepId = 0
    wolfId = 1
    xPosIndex = 2
    numXPosEachAgent = 2

    getSheepXPos = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

    # isTerminal
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    # transit for multiplayer transition; sheepTransit for sheep's MCTS simulation (only takes sheep's action as input)
    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    randomWolfPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    # reward function
    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    getActionPrior = GetActionPrior(actionSpace)

    # initialize children; expand
    initializeChildren = InitializeChildren(actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    # select next action
    selectNextAction = SelectNextAction(sheepTransit)

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # All agents' policies
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)
    policy = lambda state: [mcts(state), randomWolfPolicy(state)]

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    sampleTrajectorySingleAgentActions = SampleTrajectorySingleAgentActions(sampleTrajectory, sheepId)

    numTrials = 100
    saveDirectory = "../data/testNNPriorMCTSMujoco/trajectories"
    extension = 'pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit, 'numSimulations': numSimulations,
                          'numTrials': numTrials}
    path = getSavePath(conditionVariables)
    generateData(sampleTrajectorySingleAgentActions, policy, actionSpace, numTrials, path)


if __name__ == "__main__":
    main()