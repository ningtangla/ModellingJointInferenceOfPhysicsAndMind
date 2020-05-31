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
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, saveVariables, ApproximatePolicy
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
        grp.plot(ax=axForDraw, title='agentId={}'.format(agentId), y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))

def main():
    # manipulated variables (and some other parameters that are commonly varied)
    manipulatedVariables = OrderedDict()
    manipulatedVariables['selfIteration'] = [0, 4000,8000,16000,20000]#list(range(0,6001,6000))
    manipulatedVariables['otherIteration'] = [0, 4000,8000,16000,20000]#list(range(0,6001,6000))
    manipulatedVariables['selfId'] = [0]

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

    maxRunningSteps = 150

    sheepAliveBonus = 1 / maxRunningSteps
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
    sharedWidths = [128, 128, 128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    trainMaxRunningSteps = 30
    trainNumSimulations = 200
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgentObstacle', 'NNModel')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    depth = 4
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for agentId in range(numAgents)]
    for agentId  in range(numAgents):
        modelPath = getNNModelSavePath({'iterationIndex':-1,'agentId':agentId})
        saveVariables(multiAgentNNmodel[agentId], modelPath)

    generateTrajectoriesCodeName = 'generateMultiAgentEvaluationTrajectoryObstacle.py'
    evalNumTrials = 18
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials,
            numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    # toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgentObstacle', 'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

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
    fig = plt.figure()
    numColumns = len(manipulatedVariables['selfId'])
    numRows = 1
    plotCounter = 1

    for selfId, grp in statisticsDf.groupby('selfId'):
        grp.index = grp.index.droplevel('selfId')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylim(-1, 1)
        plt.ylabel('Accumulated rewards')
        drawPerformanceLine(grp, axForDraw, selfId)
        plotCounter += 1

    # plt.title('iterative training in chasing task with killzone radius = 2 and numSim = 200\nStart with random model')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    main()
