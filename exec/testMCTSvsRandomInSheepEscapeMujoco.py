import sys
sys.path.append('../src/algorithms')
sys.path.append('../src/sheepWolf')
sys.path.append('../src')
import numpy as np
import pandas as pd
import pylab as plt
from matplotlib import pyplot as plt
from collections import OrderedDict
import skvideo.io
import os
import pickle

skvideo.setFFmpegPath("/usr/local/bin")

# Local import
from mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, HeuristicDistanceToTarget
import reward
from wolfPoliciesFixed import PolicyActionDirectlyTowardsOtherAgent
from wrapperFunctions import GetAgentPosFromTrajectory, GetAgentPos, GetAgentActionFromTrajectoryDf, GetEpisodeLength, GetTrialTrajectoryFromDf
from envMujoco import Reset, TransitionFunction, IsTerminal
from play import SampleTrajectory


def drawPerformanceLine(dataDf, axForDraw, title):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, title=title, y='mean', yerr='std')


class GenerateTrajectories:
    def __init__(self, getSampleTrajectory, getSheepPolicies, wolfPolicy, numTrials, getSavePath):
        self.getSampleTrajectory = getSampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.numTrials = numTrials
        self.getSavePath = getSavePath
        self.wolfPolicy = wolfPolicy

    def __call__(self, oneConditionDf):
        qPosInit = oneConditionDf.index.get_level_values('qPosInit')[0]
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]

        print('qPosInit', qPosInit, 'sheepPolicy', sheepPolicyName, 'numSimulations', numSimulations)

        sampleTrajectory = self.getSampleTrajectory(qPosInit)
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]

        saveFileName = self.getSavePath(oneConditionDf)
        pickle_in = open(saveFileName, 'wb')
        pickle.dump(allTrajectories, pickle_in)
        pickle_in.close()

        return allTrajectories


def main():
    # experiment conditions
    maxRunningSteps = 3
    numTrials = 3
    manipulatedVariables = OrderedDict()
    manipulatedVariables['qPosInit'] = [(0.3, 0, -0.3, 0), (9.75, 0, 9.15, 0), (9.75, 9.75, 9.3, 9.3)]
    manipulatedVariables['sheepPolicyName'] = ['mcts', 'random']
    manipulatedVariables['numSimulations'] = [5]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # reset function
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0

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

    # wolf policies
    wolfActionMagnitude = 5
    wolfPolicyDirectlyTowardsSheep = PolicyActionDirectlyTowardsOtherAgent(getSheepXPos, getWolfXPos, wolfActionMagnitude)

    # transit for multiplayer transition; sheepTransit for sheep's MCTS simulation (only takes sheep's action as input)
    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, wolfPolicyDirectlyTowardsSheep(state)])

    # reward function
    aliveBonus = 0.05
    deathPenalty = -1
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


    getMCTS = lambda numSimulations: MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)
    getRandom = lambda numSimulations: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    sheepPolicies = {'mcts': getMCTS, 'random': getRandom}

    # wrapper function sample trajectory
    getSampleTrajectory = lambda qPosInit: SampleTrajectory(maxRunningSteps, transit, isTerminal, Reset(envModelName,
                                                        qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise))

    # function to generate and save trajectories
    dataDirectory = '../data/testMCTSvsRandomInSheepEscape/trajectories'
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)

    extension = 'pickle'
    getSavePath = GetSavePath(dataDirectory, extension)

    generateTrajectories = GenerateTrajectories(getSampleTrajectory, sheepPolicies, wolfPolicyDirectlyTowardsSheep,
                                                numTrials, getSavePath)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # function to compute statistics
    loadTrajectories = LoadTrajectories(getSavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, numTrials, len)

    # compute Statistics
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()

    plotRowNum = 1
    plotColNum = 3
    plotCounter = 1

    for (key, dataDf) in statisticsDf.groupby('qPosInit'):
        dataDf.index = dataDf.index.droplevel('qPosInit')
        axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
        drawPerformanceLine(dataDf, axForDraw, str(key))
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()


if __name__ == "__main__":
    main()


