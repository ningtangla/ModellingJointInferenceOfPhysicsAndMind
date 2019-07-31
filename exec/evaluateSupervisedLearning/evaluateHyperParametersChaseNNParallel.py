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


def drawPerformanceLine(dataDf, axForDraw, deth):
    for learningRate, grp in dataDf.groupby('learningRate'):
        grp.index = grp.index.droplevel('learningRate')
        grp.plot(ax=axForDraw, label='lr={}'.format(learningRate), y='mean', yerr='std',
                 marker='o', logx=False)


def main():
    # important parameters
    wolfId = 1

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = [64, 128, 256, 512]
    manipulatedVariables['learningRate'] = [1e-2, 1e-3, 1e-4, 1e-5]
    manipulatedVariables['trainSteps'] = [0, 20000, 40000, 60000, 80000, 100000]
    manipulatedVariables['depth'] = [2, 4, 6, 8]

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
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)

# generate trajectory parallel
    generateTrajectoriesCodeName = 'generateWolfEvaluationTrajectory.py'
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
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'evaluateSupervisedLearning', 'evaluateTrajectories')

    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trainMaxRunningSteps = 20
    trainNumSimulations = 100
    killzoneRadius = 2
    trajectoryFixedParameters = {'agentId': wolfId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the results
    fig = plt.figure()
    numColumns = len(manipulatedVariables['miniBatchSize'])
    numRows = len(manipulatedVariables['depth'])
    plotCounter = 1

    for miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')

        for depth, group in grp.groupby('depth'):
            group.index = group.index.droplevel('depth')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numRows == 1:
                axForDraw.set_ylabel('miniBatchSize: {}'.format(miniBatchSize))
            if plotCounter <= numColumns:
                axForDraw.set_title('depth: {}'.format(depth))

            # axForDraw.set_ylim(1.4, 2.5)
            # plt.ylabel('Distance between optimal and actual next position of sheep')

            drawPerformanceLine(group, axForDraw, depth)
            trainStepsLevels = statisticsDf.index.get_level_values('trainSteps').values
            axForDraw.plot(trainStepsLevels, [0.5409] * len(trainStepsLevels), label='mctsTrainData')
            plotCounter += 1

    plt.suptitle('ChaseNN Policy Accumulate Rewards')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()