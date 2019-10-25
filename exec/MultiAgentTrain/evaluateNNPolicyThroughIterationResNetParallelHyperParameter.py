import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from subprocess import Popen, PIPE
import json
import math
from collections import OrderedDict
import pickle
import pandas as pd
import time
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
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
        grp.plot(ax=axForDraw, title='selfId={}'.format(agentId), y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))

def main():
    # manipulated variables (and some other parameters that are commonly varied)
    manipulatedVariables = OrderedDict()
    manipulatedVariables['selfIteration'] = [0,300,600,900]#list(range(0,10001,2000))
    manipulatedVariables['otherIteration'] = [-999]#[-999]+list(range(0,10001,2000))
    manipulatedVariables['numTrainStepEachIteration'] = [1]
    manipulatedVariables['numTrajectoriesPerIteration'] = [1,10]
    selfId=0
    # manipulatedVariables['selfId'] = [0]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    numAgents = 2
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    sheepAliveBonus = 0.05
    wolfAlivePenalty = -sheepAliveBonus
    sheepTerminalPenalty = -1
    wolfTerminalReward = 1

    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
    rewardMultiAgents = [rewardSheep, rewardWolf]

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [256]
    actionLayerWidths = [256]
    valueLayerWidths = [256]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    trainMaxRunningSteps = 25
    trainNumSimulations = 100
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgentResNet', 'NNModelRes')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    depth = 17
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in range(numAgents)]

    for agentId  in range(numAgents):
        modelPath = getNNModelSavePath({'iterationIndex':-1,'agentId':agentId})
        saveVariables(multiAgentNNmodel[agentId], modelPath)

    generateTrajectoriesCodeName = 'generateMultiAgentResNetEvaluationTrajectoryHyperParameter.py'
    evalNumTrials = 1000
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials,
            numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgentResNet', 'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda  df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)
    measurementFunction = lambda trajectory: accumulateMultiAgentRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)
  # plot the results


    # plot the results
    fig = plt.figure()
    numColumns = len(manipulatedVariables['numTrainStepEachIteration'])
    numRows = len(manipulatedVariables['numTrajectoriesPerIteration'])
    plotCounter = 1

    for numTrainStepEachIteration, grp in statisticsDf.groupby('numTrainStepEachIteration'):
        grp.index = grp.index.droplevel('numTrainStepEachIteration')

        for numTrajectoriesPerIteration, group in grp.groupby('numTrajectoriesPerIteration'):
            group.index = group.index.droplevel('numTrajectoriesPerIteration')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numRows == 1:
                axForDraw.set_ylabel('numTrainStepEachIteration: {}'.format(numTrainStepEachIteration))
            if plotCounter <= numColumns:
                axForDraw.set_title('numTrajectoriesPerIteration: {}'.format(numTrajectoriesPerIteration))

            axForDraw.set_ylim(-1, 1.5)
            plt.ylabel('Accumulated rewards')
            drawPerformanceLine(group, axForDraw, selfId)
            plotCounter += 1



    plt.suptitle('SheepNNResnet')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
