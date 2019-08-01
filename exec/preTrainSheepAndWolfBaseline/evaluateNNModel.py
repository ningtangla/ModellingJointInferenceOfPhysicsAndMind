import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.evaluationFunctions import ComputeStatistics


def drawPerformanceLine(dataDf, axForDraw):
    dataDf.plot(ax=axForDraw, y='mean', yerr='std',
                 marker='o', logx=False)


def main():
    # important parameters
    sheepId = 0

    # manipulated variables
    manipulatedVariables = OrderedDict()
    # manipulatedVariables['miniBatchSize'] = [64, 128, 256, 512]
    # manipulatedVariables['learningRate'] = [1e-2, 1e-3, 1e-4, 1e-5]
    manipulatedVariables['trainSteps'] =[0, 20000, 40000, 60000, 80000, 100000]
    # manipulatedVariables['depth'] = [2, 4, 6, 8]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 0.05
    playDeathPenalty = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)


# generate trajectory parallel
    generateTrajectoriesCodeName = 'generateSheepEvaluationTrajectory.py'
    evalNumTrials = 1000
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.5 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName,
                                                                evalNumTrials, numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'preTrainBaseline', 'evaluateTrajectories')

    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trainMaxRunningSteps = 20
    trainNumSimulations = 200
    killzoneRadius = 2
    trajectoryFixedParameters = {'agentId': sheepId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print(statisticsDf)

    # plot the results
    fig = plt.figure()
    numColumns = 1
    numRows =   1
    plotCounter = 1

    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    # if plotCounter % numRows == 1:
    axForDraw.set_xlabel('trainingSteps: {}'.format(manipulatedVariables['trainSteps']))
    # if plotCounter <= numColumns:
    #     axForDraw.set_title('chooseActionInPlay: {}'.format(statisticsDf))

    # axForDraw.set_ylim(-1, 1)
    # plt.ylabel('Distance between optimal and actual next position of sheep')
    drawPerformanceLine(statisticsDf, axForDraw)
    plt.suptitle('EscapeNN Policy Accumulate Rewards')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
