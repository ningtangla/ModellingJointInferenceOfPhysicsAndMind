import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..','..'))

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

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories,AccumulateMultiAgentRewards

from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.evaluationFunctions import ComputeStatistics
def drawPerformanceLine(dataDf, axForDraw, agentId):
    for key, grp in dataDf.groupby('otherIteration'):
        grp.index = grp.index.droplevel('otherIteration')
        grp['agentMean'] = np.array([value[agentId] for value in grp['mean'].values])
        grp['agentStd'] = np.array([value[agentId] for value in grp['std'].values])
        grp.plot(ax=axForDraw, y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))

def main():
    # manipulated variables (and some other parameters that are commonly varied)
    manipulatedVariables = OrderedDict()
    manipulatedVariables['selfIteration'] = [0,250,450]#list(range(0,10001,2000))
    manipulatedVariables['otherIteration'] = [0,250,450]#[-999]+list(range(0,10001,2000)),
    manipulatedVariables['numTrainStepEachIteration'] = [4]
    manipulatedVariables['numTrajectoriesPerIteration'] = [16]
    selfId=1

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    
    trainMaxRunningSteps = 30
    trainNumSimulations = 200
    killzoneRadius = 2
    
    numAgents = 2
    sheepId = 0
    wolfId = 1
    posIndex = [2, 3]
 

    getSheepXPos = GetAgentPosFromState(sheepId, posIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, posIndex)

    isTerminal = IsTerminal(killzoneRadius,getWolfXPos, getSheepXPos)

    playMaxRunningSteps=50
    sheepAliveBonus = 1/playMaxRunningSteps
    wolfAlivePenalty = -sheepAliveBonus
    sheepTerminalPenalty = -1
    wolfTerminalReward = 1

    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
    rewardMultiAgents = [rewardSheep, rewardWolf]

   

    generateTrajectoriesCodeName = 'generateMultiAgentEvaluationTrajectoryObstacle.py'
    evalNumTrials = 500
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName, evalNumTrials,numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)
# 
    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName,  '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentObstacle', 'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

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
    numRows = len(manipulatedVariables['numTrainStepEachIteration'])
    numColumns = len(manipulatedVariables['numTrajectoriesPerIteration'])
    plotCounter = 1

    for numTrainStepEachIteration, grp in statisticsDf.groupby('numTrainStepEachIteration'):
        grp.index = grp.index.droplevel('numTrainStepEachIteration')

        for numTrajectoriesPerIteration, group in grp.groupby('numTrajectoriesPerIteration'):
            group.index = group.index.droplevel('numTrajectoriesPerIteration')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns==1:
                axForDraw.set_ylabel('numTrainStepEachIteration: {}'.format(numTrainStepEachIteration))
            if plotCounter <= numColumns:
                axForDraw.set_title('numTrajectoriesPerIteration: {}'.format(numTrajectoriesPerIteration))

            axForDraw.set_ylim(-1, 1.5)
            drawPerformanceLine(group, axForDraw, selfId)
            plotCounter += 1



    plt.suptitle('IterativeWolfPhysicsWithObstacle')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
