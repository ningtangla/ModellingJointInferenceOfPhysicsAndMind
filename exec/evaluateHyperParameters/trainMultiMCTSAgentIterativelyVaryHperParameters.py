import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from subprocess import Popen, PIPE
import json
import math
from collections import OrderedDict
import pandas as pd
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from matplotlib import pyplot as plt
from src.constrainedChasingEscapingEnv.envMujoco import  IsTerminal
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle
from exec.evaluationFunctions import ComputeStatistics
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete

class TrainMultiMCTSAgentParallel:
    def __init__(self, codeFileName):
        self.codeFileName = codeFileName

    def __call__(self, hyperParameterConditionslist):
        [print(condition) for condition in hyperParameterConditionslist]
        cmdList = [['python3', self.codeFileName, json.dumps(condition)]
                for condition in hyperParameterConditionslist]

        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.wait()
        return cmdList
def drawPerformanceLine(dataDf, axForDraw):
    for learningRate, grp in dataDf.groupby('learningRate'):
        grp.index = grp.index.droplevel('learningRate')
        grp.plot(ax=axForDraw, label='learningRate={}'.format(learningRate), y='mean', yerr='std',
                 marker='o', logx=False)

def main():
    # Mujoco environment
    print('start')
    manipulatedHyperVariables = OrderedDict()
    manipulatedHyperVariables['miniBatchSize'] = [64, 32]  # [64, 128, 256]
    manipulatedHyperVariables['learningRate'] = [1e-3, 1e-5]  # [1e-2, 1e-3, 1e-4]
    manipulatedHyperVariables['numSimulations'] = [5,10] #[50, 100, 200]

    #numSimulations = manipulatedHyperVariables['numSimulations']
    levelNames = list(manipulatedHyperVariables.keys())
    levelValues = list(manipulatedHyperVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    hyperVariablesConditionlist=[]
    hyperVariablesConditionlist=[{levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
    # for modelIndexNumber in range(len(modelIndex)):
    #     oneCondition={levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) for levelName in levelNames}
    #     hyperVariablesConditionlist.append(oneCondition)
    print(hyperVariablesConditionlist)
    numTrajectoriesToStartTrain = 4 * 256

    #generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMCTSWolfTrajectory.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)

    print('StratParallelGenerate')
    for numSimulations in  manipulatedHyperVariables['numSimulations']:
        trajectoryBeforeTrainPathParamters = {'iterationIndex': 0,'numSimulations':numSimulations}
        preTrainCmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
        print(preTrainCmdList)

    trainOneConditionFileName='trainMultiMCTSforOneCondition.py'
    trainMultiMCTSAgentParallel=TrainMultiMCTSAgentParallel(trainOneConditionFileName)
    trainCmdList=trainMultiMCTSAgentParallel(hyperVariablesConditionlist)
    print(trainCmdList)



    # Evaluate Session
    evluateVariables=manipulatedHyperVariables.copy()
    evluateVariables['iterationIndex']=[0,5,10]
    evluateLevelNames = list(evluateVariables.keys())
    evluatelevelValues = list(evluateVariables.values())
    evaluateModelIndex = pd.MultiIndex.from_product(evluatelevelValues, names=evluateLevelNames)
    toSplitFrame = pd.DataFrame(index=evaluateModelIndex)
    evaluatelevelNames=list(evluateVariables.keys())


    #genarate trajectories
    evalNumTrials = 10
    generateTrajectoriesCodeName = 'generateTrajByNNWolfAndStationarySheepMujoco.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.5*numCpuCores)
    numToUseCores = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials,
            numToUseCores)
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(evaluatelevelNames).apply(generateTrajectoriesParallelFromDf)


    #Measurements
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
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)

    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHyperParameters','evaluateNNPolicy', 'evaluationTrajectories')
    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 20
    trajectoryFixedParameters = {'agentId': wolfId, 'maxRunningSteps': trainMaxRunningSteps,'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda  df: getTrajectorySavePath(readParametersFromDf(df))

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(evaluatelevelNames).apply(computeStatistics)

    # plot the results
    fig = plt.figure()
    numColumns = len(evluateVariables['miniBatchSize'])
    numRows = len(evluateVariables['numSimulations'])
    plotCounter = 1

    for  miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')
        for numSimulation,subgrp in grp.groupby('numSimulations'):
            subgrp.index = subgrp.index.droplevel('numSimulations')
            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)

            if plotCounter % numRows == 1:
                axForDraw.set_ylabel('miniBatchSize: {}'.format(miniBatchSize))
            if plotCounter <= numColumns:
                axForDraw.set_title('numSimulations: {}'.format(numSimulation))
            drawPerformanceLine(subgrp, axForDraw)
            # for lr,ssubgrp in subgrp.groupby('learningRate'):
            #     ssubgrp.plot(y='mean', marker='o', label=lr, ax=axForDraw)
            plotCounter += 1


    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
