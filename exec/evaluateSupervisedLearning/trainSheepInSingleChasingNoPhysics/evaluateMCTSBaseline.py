import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import itertools as it
import pathos.multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt



from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild,  backup, InitializeChildren,Expand, RollOut
import src.constrainedChasingEscapingEnv.envNoPhysics as env
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
# from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, ApproximatePolicy, restoreVariables
from src.episode import chooseGreedyAction,SampleTrajectory
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle
from exec.evaluationFunctions import ComputeStatistics


class SampleTrajectoryFixRet:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy,trialIndex):
        state = self.reset(trialIndex)
        while self.isTerminal(state):
            state = self.reset(trialIndex)
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory



def generateOneCondition(parameters):
    numTrials = 100
    numSimulations = int(parameters['numSimulations'])

    killzoneRadius = 30
    maxRunningSteps = 150

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'numTrials':numTrials}

    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'evaluateMCTSTBaseLineTajectories')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parameters)

    numOfAgent = 2
    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]

    xBoundary = [0,600]
    yBoundary = [0,600]

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)
    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

    isTerminal = IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)


    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)


    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    # sheepActionSpace.append((0,0))

    predatorPowerRatio = 2
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

    wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    def sheepTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(wolf1Policy(state))])

    # reward function
    aliveBonus = 1 / maxRunningSteps
    deathPenalty = -1
    rewardFunction = RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    initializeChildren = InitializeChildren(
        sheepActionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return sheepActionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    rolloutHeuristic = HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    maxRolloutSteps = 10

    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    policy = lambda state:[sheepPolicy(state),wolf1Policy(state)]

    np.random.seed(1447)
    initPositionList = [[env.samplePosition(xBoundary, yBoundary) for j in range(numOfAgent)]
                        for i in range(numTrials)]
    reset = env.FixedReset(initPositionList)

    # reset = env.Reset(xBoundary, yBoundary, numOfAgent)

    sampleTrajectory = SampleTrajectoryFixRet(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

    startTime = time.time()
    trajectories = [sampleTrajectory(policy, trial) for trial in range(numTrials)]
    finshedTime = time.time() - startTime

    saveToPickle(trajectories, trajectorySavePath)

    print(parameters)
    print('lenght:',np.mean([len(tra) for tra in trajectories]))
    print('timeTaken:',finshedTime)



def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] =  [100,200,400,800]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)


    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    # trainPool.map(generateOneCondition, parametersAllCondtion)

    # load data
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'evaluateMCTSTBaseLineTajectories')
    trajectoryExtension = '.pickle'


    numTrials = 100
    killzoneRadius = 30
    maxRunningSteps = 150

    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius,'numTrials':numTrials}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    sheepId = 0
    wolfId = 1

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)

    playAliveBonus = 1/maxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = killzoneRadius
    playIsTerminal = IsTerminal(getWolfPos, getSheepPos, killzoneRadius)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)


    # compute statistics on the trajectories
    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print(statisticsDf)



    # plot the results
    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    axForDraw.set_ylim(-1, 1.5)
    statisticsDf.plot(ax=axForDraw, y='mean', yerr='std',marker='o', logx=False)
    plt.ylabel('Accumulated rewards')
    plt.suptitle('Evaulate MCTS Baseline')
    plt.legend(loc='best')
    plt.show()



    def calculateSuriveRatio(trajectory):
        lenght = np.array(len(trajectory))
        count = np.array([lenght<50, lenght>=50 and lenght<100,lenght>=100])
        return count
    computeNumbers = ComputeStatistics(loadTrajectoriesFromDf, calculateSuriveRatio)
    df = toSplitFrame.groupby(levelNames).apply(computeNumbers)
    print(df)

    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    xlabel = ['0-50','50-100','100-150']
    x = np.arange(len(xlabel))

    yMean = df['mean'].tolist()
    yRrr = np.array(df['std'].tolist()) / (np.sqrt(numTrials) -1)

    totalWidth, n = 0.6, 3
    width = totalWidth / n

    x = x - (totalWidth - width) / 2
    plt.bar(x, yMean[0], yerr=yRrr[0],   width=width, label='simulation20')
    plt.bar(x + width, yMean[1], yerr=yRrr[1], width=width, label='simulation50')
    plt.bar(x + width * 2, yMean[2], yerr=yRrr[2],width=width, label='simulation100')
    plt.bar(x + width * 3, yMean[3], yerr=yRrr[3],width=width, label='simulation200')
    plt.xticks(x, xlabel)
    plt.ylim(0, 1)
    plt.xlabel('living steps')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()