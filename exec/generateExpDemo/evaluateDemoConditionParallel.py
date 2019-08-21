import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..'))
# import ipdb
import itertools as it
import numpy as np
from collections import OrderedDict, deque
import pandas as pd
from matplotlib import pyplot as plt
import mujoco_py as mujoco
import math

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import  SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel,ExcuteCodeOnConditionsParallel
from exec.evaluationFunctions import ComputeStatistics
from exec.generateExpDemo.filterTraj import *

def drawPerformanceLine(dataDf, axForDraw):
    for masterPowerRatio, grp in dataDf.groupby('masterPowerRatio'):
        grp.index = grp.index.droplevel('masterPowerRatio')
        grp.plot(ax=axForDraw, label='masterPowerRatio={}'.format(masterPowerRatio), y='mean', yerr='std', marker='o', logx=False)

def main():
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['masterPowerRatio'] = [0.2, 0.4]
    manipulatedVariables['beta'] = [0.5, 1.0, 2.0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)


    sampleTrajectoryFileName = 'sampleConditionTraj.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.88 * numCpuCores)
    numTrials = math.floor(numCpuToUse/len(parametersAllCondtion)) * 1
    generateTrajectoriesParallel = ExcuteCodeOnConditionsParallel(sampleTrajectoryFileName, numTrials, numCpuToUse)

    print("start")
    startTime = time.time()

    cmdList = generateTrajectoriesParallel(parametersAllCondtion)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


    maxRunningSteps = 360
    numSimulations = 600
    killzoneRadius = 0.5

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))


    sheepId=0
    wolfId=1
    masterId=2
    stateIndex=0
    positionIndex=[0,1]
    velocityIndex=[4,5]

    calculateChasingDeviation = CalculateChasingDeviation(sheepId, wolfId, stateIndex, positionIndex)

    countCross=CountSheepCrossRope(sheepId, wolfId, masterId,stateIndex, positionIndex,tranformCoordinates,isCrossAxis)

    angleVariance = math.pi / 10
    lowBound = math.pi / 2 - angleVariance
    upBound = math.pi/ 2 + angleVariance
    isInAngelRange = IsAllInAngelRange(lowBound, upBound)
    timeWindow=5
    countCircles=CountCirclesBetweenWolfAndMaster(wolfId, masterId,stateIndex, positionIndex, velocityIndex, timeWindow,isInAngelRange)

    spaceSize=10
    cornerSize=3
    countCorner = CountSheepInCorner(sheepId,stateIndex, positionIndex,spaceSize,cornerSize,isInCorner)

    collisionRadius = 0.8
    countCollision = CountCollision(sheepId,wolfId,stateIndex, positionIndex,collisionRadius,isCollision)

    measurementFunctionName = ['calculateChasingDeviation','countCross', 'countCircles', 'countCorner','countCollision']
    measurementFunctionList = [calculateChasingDeviation, countCross, countCircles, countCorner,countCollision]
    ylim=[[0.8, 1.6], [0, 0.1], [0, 0.2], [0, 1], [0, 0.01]] 
    numRows = 1
    numColumns = 5
    plotCounter = 1
    fig = plt.figure()
    for index in range(len(measurementFunctionList)):

        computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunctionList[index])
        statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
        print(statisticsDf)
        # plot the results

        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        plotCounter = plotCounter + 1
        drawPerformanceLine(statisticsDf, axForDraw)

        axForDraw.set_title("measure={}".format(measurementFunctionName[index]))
        axForDraw.set_ylim(ylim[index][0], ylim[index][1])


        plt.legend(loc='best')
    plt.show()
    picSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo', 'trajectories', 'pic')
    if not os.path.exists(picSaveDirectory):
        os.makedirs(picSaveDirectory)
    picFileName = "mcts=uniformPriorRollout.png"#.format(measurementFunctionName[index])
    fig.savefig(os.path.join(picSaveDirectory, picFileName))

if __name__ == '__main__':
    main()
