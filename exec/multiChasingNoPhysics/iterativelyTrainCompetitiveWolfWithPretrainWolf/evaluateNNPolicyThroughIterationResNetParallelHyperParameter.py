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
    for key, grp in dataDf.groupby('otherIteration'):
        grp.index = grp.index.droplevel('otherIteration')
        grp['agentMean'] = np.array([value[agentId] for value in grp['mean'].values])
        grp['agentStd'] = np.array([value[agentId] for value in grp['std'].values])
        grp.plot(ax=axForDraw, y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))


def main():
    # manipulated variables (and some other parameters that are commonly varied)
    manipulatedVariables = OrderedDict()
    manipulatedVariables['selfIteration'] = list(range(0, 1001, 100))
    manipulatedVariables['otherIteration'] = list(range(0, 1001, 100))
    manipulatedVariables['numTrainStepEachIteration'] = [3]
    manipulatedVariables['numTrajectoriesPerIteration'] = [16]

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
    getWolfOneXPos = GetAgentPosFromState(wolfOnePosIndex, posIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoIndex, posIndex)

    playerKillzone = killzoneRadius
    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, playerKillzone)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, playerKillzone)

    def isTerminal(state): return isTerminalOne(state) or isTerminalTwo(state)

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
    trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrain2Wolves1SheepWithPretrainModelMultiTrees', 'evaluateTrajectories')
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
    numRows = len(manipulatedVariables['numTrainStepEachIteration'])
    numColumns = len(manipulatedVariables['numTrajectoriesPerIteration'])
    plotCounter = 1

    for numTrainStepEachIteration, grp in statisticsDf.groupby('numTrainStepEachIteration'):
        grp.index = grp.index.droplevel('numTrainStepEachIteration')

        for numTrajectoriesPerIteration, group in grp.groupby('numTrajectoriesPerIteration'):
            group.index = group.index.droplevel('numTrajectoriesPerIteration')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns == 1:
                axForDraw.set_ylabel('numTrainStepEachIteration: {}'.format(numTrainStepEachIteration))
            if plotCounter <= numColumns:
                axForDraw.set_title('numTrajectoriesPerIteration: {}'.format(numTrajectoriesPerIteration))

            axForDraw.set_ylim(-1, 1.5)
            drawPerformanceLine(group, axForDraw, selfId)
            plotCounter += 1

    plt.suptitle('SheepNNResnet Step=100 Simulation=200')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
