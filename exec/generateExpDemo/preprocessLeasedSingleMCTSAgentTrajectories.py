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
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel, ExcuteCodeOnConditionsParallel
from src.constrainedChasingEscapingEnv.demoFilter import CalculateChasingDeviation, CalculateDistractorMoveDistance, OffsetMasterStates, FindCirlceBetweenWolfAndMaster


def main():
    startTime = time.time()

    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'generateExpDemo','trajectories')

    trajectoryExtension = '.pickle'

    sheepId = 0
    wolfId = 1
    masterId = 2
    distractorId = 3
    maxRunningSteps = 250
    numSimulations = 200
    killzoneRadius = 0.5

    # preyPowerRatio = 0.7
    # predatorPowerRatio = 1.3
    # masterPowerRatio = 0.3
    # distractorPowerRatio = 0.7
    # trajectoryFixedParameters = {'agentId': sheepId, 'maxRunningSteps':maxRunningSteps, 'killzoneRadius':killzoneRadius, 'numSimulations':numSimulations,\
    #         'preyPowerRatio':preyPowerRatio, 'predatorPowerRatio':predatorPowerRatio, 'masterPowerRatio':masterPowerRatio, 'distractorPoweRatio':distractorPowerRatio}
    trajectoryFixedParameters = {'agentId': 310, 'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius, 'numSimulations': numSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    pathParameters = {}
    trajectories = loadTrajectories(pathParameters)

    stateIndex = 0
    qPosIndex = [0, 1]
    calculateChasingDeviation = CalculateChasingDeviation(sheepId, wolfId, stateIndex, qPosIndex)
    trajectoryDeviationes = np.array([np.mean(calculateChasingDeviation(trajectory)) for trajectory in trajectories])
    trajectoryLengthes = np.array([len(trajectory) for trajectory in trajectories])
    calculateDistractorMoveDistance = CalculateDistractorMoveDistance(distractorId, stateIndex, qPosIndex)
    calculateSheepMoveDistance = CalculateDistractorMoveDistance(sheepId, stateIndex, qPosIndex)
    trajectoryDistractorMoveDistances = np.array([np.mean(calculateDistractorMoveDistance(trajectory)) for trajectory in trajectories])
    trajectorySheepMoveDistances = np.array([np.mean(calculateSheepMoveDistance(trajectory)) for trajectory in trajectories])

    print(len(trajectories))
    minLength = 250
    minDeviation = math.pi/400
    maxDeviation = math.pi/1

    minDistractorMoveDistance = 0
    maxDistractorMoveDistance = 100

    deviationLegelTraj = filter(lambda x: x <= maxDeviation and x >= minDeviation, trajectoryDeviationes)
    deviationLegelTrajIndex = [list(trajectoryDeviationes).index(i) for i in deviationLegelTraj]

    timeWindow = 10
    angleVariance = math.pi / 6
    circleFilter = FindCirlceBetweenWolfAndMaster(wolfId, masterId, stateIndex, qPosIndex, timeWindow, angleVariance)
    filterlist = [circleFilter(trajectory) for trajectory in trajectories]
    timewindowLeagelTrajIndex = np.where(filterlist)[0]

    lengthLeagelTrajIndex = np.nonzero(trajectoryLengthes >= minLength)
    distractorMoveDistanceLegelTraj = filter(lambda x: x <= maxDistractorMoveDistance and x >= minDistractorMoveDistance, trajectoryDistractorMoveDistances)
    distractorMoveDistanceLegelTrajIndex = [list(trajectoryDistractorMoveDistances).index(i) for i in distractorMoveDistanceLegelTraj]

    leagelTrajIndex = ft.reduce(np.intersect1d, [deviationLegelTrajIndex, lengthLeagelTrajIndex, distractorMoveDistanceLegelTrajIndex, timewindowLeagelTrajIndex])
    # leagelTrajIndex = lengthLeagelTrajIndex[0]
    print(leagelTrajIndex)
    print('leagelTraj length:', len(leagelTrajIndex))
    print('mean deviation:', np.mean(trajectoryDeviationes[leagelTrajIndex]))

    leagelTrajectories = [trajectory[:minLength] for trajectory in np.array(trajectories)[leagelTrajIndex]]
    masterDelayStep = 8
    offsetMasterStates = OffsetMasterStates(masterId, stateIndex, masterDelayStep)
    masterDelayedStates = [offsetMasterStates(trajectory) for trajectory in leagelTrajectories]
    masterDelayedStatesPathParameters = {'offset': masterDelayStep}
    masterDelayedStatesPath = getTrajectorySavePath(masterDelayedStatesPathParameters)
    saveToPickle(masterDelayedStates, masterDelayedStatesPath)

    demoStepIndex = list(range(masterDelayStep, minLength))
    demoTajectories = [np.array(trajectory)[demoStepIndex] for trajectory in leagelTrajectories]
    demoStates = np.array([[timestep[stateIndex] for timestep in trajectory] for trajectory in demoTajectories])
    demoStatesPathParameters = {'offset': 0}
    demoStatesPath = getTrajectorySavePath(demoStatesPathParameters)
    saveToPickle(demoStates, demoStatesPath)
    endTime = time.time()

    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
