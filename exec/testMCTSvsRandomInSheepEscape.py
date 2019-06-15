import sys
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
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, HeuristicDistanceToTarget
import reward
from envSheepChaseWolf import stationaryWolfPolicy, WolfPolicyForceDirectlyTowardsSheep, DistanceBetweenActualAndOptimalNextPosition, \
    GetAgentPosFromTrajectory, GetAgentPos, GetAgentActionFromTrajectoryDf, GetEpisodeLength, GetTrialTrajectoryFromDf
from envMujoco import Reset, TransitionFunction, IsTerminal
from play import SampleTrajectory


def drawPerformanceLine(dataDf, axForDraw, title):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, title=title, y = 'mean', yerr='std')


class GetSheepPolicy:
    def __init__(self, getMCTS, randomPolicy):
        self.getMCTS = getMCTS
        self.randomPolicy = randomPolicy

    def __call__(self, policyName, numSimulations):
        mcts = self.getMCTS(numSimulations)
        policies = {'mcts': mcts, 'random': self.randomPolicy}

        return policies[policyName]


class PrepareAllAgentsPolicy:
    def __init__(self, getSheepPolicy, wolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.wolfPolicy = wolfPolicy

    def __call__(self, sheepPolicyName, sheepPolicyNumSimulations):
        sheepPolicy = self.getSheepPolicy(sheepPolicyName, sheepPolicyNumSimulations)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        return policy


def tupleToString(tuple):
    string = '('

    for index in range(len(tuple)):
        string += str(tuple[index]).replace('.', ',')
        if(index != len(tuple)-1):
            string += '_'

    string += ')'

    return string


class GetSaveFileName:
    def __init__(self, dataDirectory, tupleToString):             # CHANGE THIS NAME PARENTDIRECTORY
        self.dataDirectory = dataDirectory
        self.tupleToString = tupleToString

    def __call__(self, oneConditionDf):
        modelDf = oneConditionDf.reset_index()
        manipulatedVariableNames = modelDf.columns.values.tolist()
        manipulatedVariableValues = {name: modelDf[name][0] for name in manipulatedVariableNames}

        fileName = self.dataDirectory + '/'

        for variable in manipulatedVariableValues:
            fileName += variable
            fileName += '='

            variableValue = manipulatedVariableValues[variable]
            if type(variableValue) is tuple:
                fileName += self.tupleToString(variableValue)
            else:
                fileName += str(variableValue)

            fileName += '|'

        fileName += '.pickle'

        return fileName


class ComputeMeanMeasurement:
    def __init__(self, getSaveFileName, measurementFunction):
        self.getSaveFileName = getSaveFileName
        self.measurementFunction = measurementFunction

    def __call__(self, conditionDf):
        readFile = self.getSaveFileName(conditionDf)
        trajectoriesDf = pd.read_pickle(readFile)

        trajectoriesDf['measurements'] = trajectoriesDf.apply(self.measurementFunction, axis=1)

        return pd.Series({'mean': trajectoriesDf['measurements'].mean(axis=0), 'std': trajectoriesDf['measurements'].std(axis=0)})


class GenerateTrajectoriesAndComputeStatistics:
    def __init__(self, getReset, getSampleTrajectory, prepareAllAgentsPolicy, numTrials, getSaveFileName):
        self.getReset = getReset
        self.getSampleTrajectory = getSampleTrajectory
        self.prepareAllAgentsPolicy = prepareAllAgentsPolicy
        self.numTrials = numTrials
        self.getSaveFileName = getSaveFileName

    def __call__(self, oneConditionDf):
        modelDf = oneConditionDf.reset_index()
        qPosInit = modelDf['qPosInit'][0]
        sheepPolicyName = modelDf['sheepPolicyName'][0]
        numSimulations = modelDf['numSimulations'][0]

        print('qPosInit', qPosInit, 'sheepPolicy', sheepPolicyName, 'numSimulations', numSimulations)

        reset = self.getReset(qPosInit)
        sampleTrajectory = self.getSampleTrajectory(reset)
        policy = self.prepareAllAgentsPolicy(sheepPolicyName, numSimulations)

        allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]

        allEpisodeLengths = [len(trajectory) for trajectory in allTrajectories]
        meanEpisodeLength = np.mean(allEpisodeLengths)
        episodeLengthStdDev = np.std(allEpisodeLengths)

        saveFileName = self.getSaveFileName(oneConditionDf)
        pickle_in = open(saveFileName, 'wb')
        pickle.dump(allTrajectories, pickle_in)
        pickle_in.close()

        returnSeries = pd.Series({'mean': meanEpisodeLength, 'std': episodeLengthStdDev})

        return returnSeries

def main():
    # experiment conditions
    maxRunningSteps = 15
    numTrials = 50
    manipulatedVariables = OrderedDict()
    manipulatedVariables['qPosInit'] = [(0.3, 0, -0.3, 0), (9.75, 0, 9.15, 0), (9.75, 9.75, 9.3, 9.3)]
    manipulatedVariables['sheepPolicyName'] = ['mcts', 'random']
    manipulatedVariables['numSimulations'] = [5, 25, 50, 100, 250, 400]

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

    # isTerminal
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius)

    # transit for multiplayer transition; sheepTransit for sheep's MCTS simulation (only takes sheep's action as input)
    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryWolfPolicy(state)])

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

    sheepId = 0
    wolfId = 1
    xPosIndex = 2
    numXPosEachAgent = 2

    getSheepXPos = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # wrapper function
    getReset = lambda qPosInit: Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # All agents' policies
    getMCTS = lambda numSimulations: MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)
    sheepPolicyRandom = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getSheepPolicy = GetSheepPolicy(getMCTS, sheepPolicyRandom)
    wolfActionMagnitude = 5
    wolfPolicyDirectlyTowardsSheep = WolfPolicyForceDirectlyTowardsSheep(getSheepXPos, getWolfXPos, wolfActionMagnitude)
    prepareAllAgentsPolicy = PrepareAllAgentsPolicy(getSheepPolicy, wolfPolicyDirectlyTowardsSheep)

    # sample trajectory
    getSampleTrajectory = lambda reset: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # function to generate and save trajectories
    dataDirectory = '../data/testMCTSvsRandomInSheepEscape/trajectories'
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)

    getSaveFileName = GetSaveFileName(dataDirectory, tupleToString)
    generateTrajectoriesAndComputeStatistics = GenerateTrajectoriesAndComputeStatistics(getReset, getSampleTrajectory,
                                                                                        prepareAllAgentsPolicy,
                                                                                        numTrials, getSaveFileName)

    # run all trials and save trajectories
    statisticsDf = toSplitFrame.groupby(levelNames).apply(generateTrajectoriesAndComputeStatistics)

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


