import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,IsCollided
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



def main():
    # important parameters
    distractorId = 3
    sampleOneStepPerTraj =  1# True
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['dataSize'] =  [1000,3000,5000]#[1000,3000,9000]
    manipulatedVariables['depth'] =  [4,5,9]

    manipulatedVariables['trainSteps'] =list(range(0,50001,10000))


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)



    playAliveBonus = 0.004
    playDeathPenalty = -1
    playKillzoneRadius = 20
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    # addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

# generate trajectory parallel
    generateTrajectoriesCodeName = 'generateSheepSingleChasingEvaluationTrajectoryCondition.py'
    evalNumTrials = 500
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName,evalNumTrials, numCmdList)

    # run all trials and save trajectories
    # generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    # toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateEscapeSingleChasingNoPhysics', 'evaluateSheepTrajectoriesHyperParameter')

    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'


    killzoneRadius = 20
    numSimulations = 200 #100
    maxRunningSteps = 250
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'sampleOneStepPerTraj':0}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # # plot the results
    # fig = plt.figure()
    # numRows = 1
    # numColumns = 1
    # plotCounter = 1
    # axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    # statisticsDf.plot(ax=axForDraw, y='mean', yerr='std',marker='o', logx=False)
    # plt.suptitle('policyValueNet')
    # plt.legend(loc='best')
    # plt.show()

    # plot the results
    fig = plt.figure()
    numColumns = len(manipulatedVariables['depth'])
    numRows = len(manipulatedVariables['dataSize'])
    plotCounter = 1
    print(statisticsDf)
    for depth, grp in statisticsDf.groupby('depth'):
        grp.index = grp.index.droplevel('depth')

        for dataSize, group in grp.groupby('dataSize'):
            group.index = group.index.droplevel('dataSize')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numRows == 1:
                axForDraw.set_ylabel('depth: {}'.format(depth))
            if plotCounter <= numColumns:
                axForDraw.set_title('dataSize: {}'.format(dataSize))

            axForDraw.set_ylim(-1, 0)
            # plt.ylabel('Accumulated rewards')
            maxTrainSteps = manipulatedVariables['trainSteps'][-1]

            plt.plot([0, maxTrainSteps], [-0.76]*2, '--m', color="#1C2833", label='pure MCTS')

            group.plot(ax=axForDraw, y='mean', yerr='std',marker='o', logx=False)

            plotCounter += 1



    plt.suptitle('SheepNN noSample')
    plt.legend(loc='best')
    plt.show()

def drawPerformanceLine(dataDf, axForDraw, agentId):
    for key, grp in dataDf.groupby('otherIteration'):
        grp.index = grp.index.droplevel('otherIteration')
        grp['agentMean'] = np.array([value[agentId] for value in grp['mean'].values])
        grp['agentStd'] = np.array([value[agentId] for value in grp['std'].values])
        grp.plot(ax=axForDraw, title='agentId={}'.format(agentId), y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))

if __name__ == '__main__':
    main()
