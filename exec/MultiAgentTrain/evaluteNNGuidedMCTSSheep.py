import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..'))
# import ipdb

import numpy as np
from collections import OrderedDict, deque
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import  SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel



def main():
    dirName = os.path.dirname(__file__)
    # load save dir
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'multiAgentTrain', 'multiMCTSAgent', 'NNGuideMCTSSheepTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    maxRunningSteps = 20
    numSimulations = 200
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'

    sheepId = 0
    wolfId = 1

    startTime = time.time()

    numTrajectories = 1000
    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleNNGuidedMCTSSheep.py'
    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.75*numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectories, numCmdList)
    pathParameters = {'iterationIndex':-1}
    cmdList = generateTrajectoriesParallel(pathParameters)


    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)    
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
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
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)


    trajectories = loadTrajectories(pathParameters)

    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
 
    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
    std = np.std([tra[0][3] for tra in valuedTrajectories])

    print(trainDataMeanAccumulatedReward, std)

if __name__ == '__main__':
    main()