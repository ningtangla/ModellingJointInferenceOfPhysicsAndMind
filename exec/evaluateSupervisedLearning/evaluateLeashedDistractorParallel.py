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
<<<<<<< HEAD
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, IsCollided, RewardFunctionWithWall
=======
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,IsCollided
>>>>>>> 2eb18c7353c19461cedd0166042f47cad81a5ea0
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

class RewardFunctionForDistractor():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isTerminal, getPosition, getVelocity):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.velocityBound = velocityBound
        self.isTerminal = isTerminal
        self.getPosition = getPosition
        self.getVelocity = getVelocity

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish =  - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0,self.safeBound -  minDisToWall), 2) / np.power(self.safeBound, 2)

        agentVel = self.getVelocity(state)
        velPunish = -np.abs(self.aliveBonus) if np.linalg.norm(agentVel) <= self.velocityBound else 0

        return reward + wallPunish + velPunish

def main():
    # important parameters
    distractorId = 3

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = [128, 256]
    manipulatedVariables['learningRate'] =  [1e-3, 1e-4]
    manipulatedVariables['depth'] = [4]
    manipulatedVariables['trainSteps'] = list(range(0,100001, 20000))


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    masterId = 2
    distractorId = 3
    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    getMasterPos = GetAgentPosFromState(masterId, xPosIndex)
    getDistractorPos = GetAgentPosFromState(distractorId, xPosIndex)
    killzoneRadius = 1

    killzoneRadius = 1
    aliveBonus = 0.05
    deathPenalty = -1
    safeBound = 2.5
    wallDisToCenter = 10
    wallPunishRatio = 4
    velIndex = [4, 5]
    getDistractorVel =  GetAgentPosFromState(distractorId, velIndex)
    velocityBound = 5

    otherIds = list(range(13))
    otherIds.remove(distractorId)
    getOthersPos = [GetAgentPosFromState(otherId, xPosIndex) for otherId in otherIds]

    isCollided = IsCollided(killzoneRadius, getDistractorPos, getOthersPos)
    playReward = RewardFunctionForDistractor(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isCollided, getDistractorPos, getDistractorVel)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)

# generate trajectory parallel
    generateTrajectoriesCodeName = 'generateLeashedDistractorEvaluationTrajectory.py'
    evalNumTrials = 500
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName,evalNumTrials, numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    # toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'evaluateSupervisedLearning', 'evaluateLeashedDistractorTrajectories')

    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'

    trainMaxRunningSteps = 25
    trainNumSimulations = 200
    killzoneRadius = 1
    trajectoryFixedParameters = {'agentId': distractorId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

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
    numRows = len(manipulatedVariables['miniBatchSize'])
    numColumns = len(manipulatedVariables['depth'])
    plotCounter = 1

    for miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')

        for depth, group in grp.groupby('depth'):
            group.index = group.index.droplevel('depth')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel('miniBatchSize: {}'.format(miniBatchSize))
            if plotCounter <= numColumns:
                axForDraw.set_title('depth: {}'.format(depth))

            # axForDraw.set_ylim(-1, 0.6)
            # plt.ylabel('Distance between optimal and actual next position of sheep')

            drawPerformanceLine(group, axForDraw, depth)
            trainStepsLevels = statisticsDf.index.get_level_values('trainSteps').values
            axForDraw.plot(trainStepsLevels, [0.193] * len(trainStepsLevels), label='mctsTrainData')
            plotCounter += 1

<<<<<<< HEAD
    plt.suptitle('DistractorNN Policy Accumulate Rewards')
=======
    plt.suptitle('Distractor NN Policy Accumulate Rewards')
>>>>>>> 2eb18c7353c19461cedd0166042f47cad81a5ea0
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
