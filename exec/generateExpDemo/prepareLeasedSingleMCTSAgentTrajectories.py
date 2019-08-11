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
import itertools as it
import functools as ft
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
from exec.parallelComputing import GenerateTrajectoriesParallel, ExcuteCodeOnConditionsParallel
from src.constrainedChasingEscapingEnv.demoFilter import CalculateChasingSubtlety, CalculateDistractorMoveDistance


def main():
    startTime = time.time()

    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning',
                                       'leashedSheepTrajectories')

    trajectoryExtension = '.pickle'

    sheepId = 0
    wolfId = 1
    masterId = 2
    distractorId = 3
    maxRunningSteps = 120
    numSimulations = 200
    trajectoryFixedParameters = {'agentId': sheepId, 'maxRunningSteps':maxRunningSteps, 'numSimulations':numSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    pathParameters = {}
    trajectories = loadTrajectories(pathParameters)

    stateIndex = 0
    qPosIndex = [0, 1]
    calculateChasingSubtlety = CalculateChasingSubtlety(sheepId, wolfId, stateIndex, qPosIndex)
    trajectorySubtleties = np.array([np.mean(calculateChasingSubtlety(trajectory)) for trajectory in trajectories])
    trajectoryLengthes = np.array([len(trajectory) for trajectory in trajectories])
    trajectoryDistractorMoveDistances = np.array([np.mean(CalculateDistractorMoveDistance(trajectory)) for trajectory in trajectories])
    minLength = 90
    minSubtlety = math.pi/4
    minDistractorMoveDistance = 0.2
    subtletyLegelTrajIndex = np.nonzero(trajectorySubtleties >= minSubtlety)
    lengthLeagelTrajIndex = np.nonzero(trajectoryLengthes >= minLength)
    distractorMoveDistanceLegelTrajIndex = np.nonzero(trajectoryDistractorMoveDistances >= minDistractorMoveDistance)

    leagelTrajIndex = reduce(np.intersect1d, [subtletyLegelTrajIndex, lengthLeagelTrajIndex, distractorMoveDistanceLegelTrajIndex])
    
    leagelTrajectories = trajectories[leagelTrajIndex]
    leagelTrajectoriesPathParameters = {'offset': 0}
    leagelTrajectoriesPath = getTrajectorySavePath(leagelTrajectoriesPathParameters)
    saveToPickle(leagelTrajectories, leagelTrajectoriesPathParameters)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()

