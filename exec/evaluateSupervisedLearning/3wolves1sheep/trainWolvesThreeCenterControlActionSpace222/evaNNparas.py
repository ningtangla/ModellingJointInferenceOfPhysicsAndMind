import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
from collections import OrderedDict
import pickle
import pandas as pd
import time
from matplotlib import pyplot as plt
import numpy as np
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryByReflectVelocity

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist, RollOut
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards, AccumulateMultiAgentRewards
from exec.parallelComputing import GenerateTrajectoriesParallel


def drawPerformanceLine(dataDf, axForDraw, agentId):
    for key, grp in dataDf.groupby('isDataAugmented'):
        grp.index = grp.index.droplevel('isDataAugmented')
        grp['agentMean'] = np.array([value[agentId] for value in grp['mean'].values])
        grp['agentStd'] = np.array([value[agentId] for value in grp['std'].values])
        grp.plot(ax=axForDraw, y='agentMean', yerr='agentStd', marker='o', label='isDataAugmented={}'.format(key))


def main():
    # manipulated variables (and some other parameters that are commonly varied)
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = list(range(0, 100001, 10000))
    manipulatedVariables['isDataAugmented'] = [0, 1]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    trainMaxRunningSteps = 50
    trainNumSimulations = 250
    killzoneRadius = 50

    sheepId = 0
    wolfId = 1
    posIndex = [0, 1]
    selfId = sheepId

    wolfOnePosIndex = 1
    wolfTwoIndex = 2
    getSheepXPos = GetAgentPosFromState(sheepId, posIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, posIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, posIndex)
    getWolfThreeXPos = GetAgentPosFromState(wolfThreeId, posIndex)

    playerKillzone = killzoneRadius
    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    isTerminalThree = IsTerminal(getWolfThreeXPos, getSheepXPos, killzoneRadius)

    isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state) or isTerminalThree(state)

    sheepAliveBonus = 1 / trainMaxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus
    sheepTerminalPenalty = -1
    wolfTerminalReward = 1

    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
    rewardMultiAgents = [rewardSheep, rewardWolf]

    generateTrajectoriesCodeName = 'generateMultiAgentResNetEvaluationTrajectoryHyperParameter.py'
    evalNumTrials = 500
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials, numCmdList)

    # run all trials and save trajectories

    def generateTrajectoriesParallelFromDf(df): return generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = oos.path.join(dirName, '..', '..', '..', '..', 'data', '3wolves1sheep', 'trainWolvesThreeCenterControlAction444', 'evaTraj')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    def loadTrajectoriesFromDf(df): return loadTrajectories(readParametersFromDf(df))

    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

    def measurementFunction(trajectory): return accumulateMultiAgentRewards(trajectory)[0]

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)

    # plot the results
    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1

    for trainSteps, group in grp.groupby('trainSteps'):
        group.index = group.index.droplevel('trainSteps')

        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        # if (plotCounter % numColumns == 1) or numColumns == 1:
        # axForDraw.set_ylabel('numTrainStepEachIteration: {}'.format(numTrainStepEachIteration))
        # if plotCounter <= numColumns:
        #     axForDraw.set_title('trainSteps: {}'.format(trainSteps))

        axForDraw.set_ylim(-1, 1.5)
        drawPerformanceLine(group, axForDraw, selfId)
        plotCounter += 1

    plt.suptitle('wolfNNResnet')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()