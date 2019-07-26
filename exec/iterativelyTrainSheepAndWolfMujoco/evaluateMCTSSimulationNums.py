import sys
import os
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
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards


def drawPerformanceLine(dataDf, axForDraw, numSimulations, trainSteps):
    for key, grp in dataDf.groupby('modelLearnRate'):
        grp.index = grp.index.droplevel('modelLearnRate')
        grp.plot(ax=axForDraw, title='numSimulations={}, trainSteps={}'.format(numSimulations, trainSteps), y='mean', yerr='std', marker='o', label='learnRate={}'.format(key))


class GenerateTrajectoriesParallel:
    def __init__(self, codeFileName, numSample, numCmdList, readParametersFromDf):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList
        self.readParametersFromDf = readParametersFromDf

    def __call__(self, oneConditionDf):
        startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample/self.numCmdList))
        endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
        startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
        parameters = self.readParametersFromDf(oneConditionDf)
        parametersString = dict([(key, str(value)) for key, value in parameters.items()])
        parametersStringJS = json.dumps(parametersString)
        cmdList = [['python3', self.codeFileName, parametersStringJS, str(startSampleIndex), str(endSampleIndex)] 
                for startSampleIndex, endSampleIndex in startEndIndexesPair]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.wait()
        return cmdList

def main():
    # manipulated variables (and some other parameters that are commonly varied)
    evalNumSimulations = 20  # 200
    evalNumTrials = 2  # 1000
    evalMaxRunningSteps = 3
    manipulatedVariables = OrderedDict()
   
    manipulatedVariables['iteration'] = [0, 10, 20]
    manipulatedVariables['numSimulations'] =[50] #E[50, 100, 150, 200]
    manipulatedVariables['policyName'] = ['NNPolicy']  # ['NNPolicy', 'mctsHeuristic']

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)
    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    generateTrajectoriesCodeName = 'generateTrajByNNWolfAndStationarySheepMujoco.py'
    numToUseCores = min(evalNumTrials, 4)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials,
            numToUseCores, readParametersFromDf)
    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallel)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'iterativelyTrainMultiAgent',
                                       'evaluateNNPolicy', 'evaluationTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trainMaxRunningSteps = 25
    #trainNumSimulations = 20
    trajectoryFixedParameters = {'agentId': wolfId, 'maxRunningSteps': trainMaxRunningSteps,'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda  df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    for policyName, grp in statisticsDf.groupby('policyName'):
        
        for numSimulations,subgrp in grp.groupby('numSimulations'):
            subgrp.index = subgrp.index.droplevel('numSimulations')
            subgrp.plot(y='mean', marker='o', label=numSimulations, ax=axis)

    plt.ylabel('Accumulated rewards')
    plt.title('iterative training in chasing task with killzone radius = 2 and numSim = 200\nStart with random model')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
