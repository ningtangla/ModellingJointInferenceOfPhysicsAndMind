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
from src.constrainedChasingEscapingEnv.demoFilter import OffsetMasterStates, FindCirlceBetweenWolfAndMaster
from exec.generateExpDemo.filterTraj import CalculateChasingDeviation, CalculateDistractorMoveDistance, CountSheepCrossRope, isCrossAxis, tranformCoordinates, CountCollision, isCollision


def main():
    startTime = time.time()

    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'generateExpDemo','trajectories')

    trajectoryExtension = '.pickle'

    sheepId = 0
    wolfId = 1
    masterId = 2
    distractorId = 3
    maxRunningSteps = 360
    numSimulations = 400
    killzoneRadius = 0.5

    manipulatedVariables = OrderedDict()
    manipulatedVariables['beta'] = [1.0]
    manipulatedVariables['masterPowerRatio'] = [0.4]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditionParametersAll = [dict(list(i)) for i in productedValues]

    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius, 'numSimulations': numSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    masterDelayStep = 6
    minLength = 250 + masterDelayStep
    timestepCheckInterval = 40

    for pathParameters in conditionParametersAll:
        originalTrajectories = loadTrajectories(pathParameters)
        numRunningStepsInOrigTra = np.array([len(trajectory) for trajectory in originalTrajectories])
        print(numRunningStepsInOrigTra, pathParameters)
        leagelOriginalTrajctories = np.array(originalTrajectories)[np.nonzero(numRunningStepsInOrigTra >= minLength)]
        if len(leagelOriginalTrajctories) > 0:
            maxCheckingTimeStep = np.min([len(trajectory) for trajectory in leagelOriginalTrajctories])
            checkTimesteps = np.arange(0, maxCheckingTimeStep - minLength, timestepCheckInterval)

            for startCheckTimestep in checkTimesteps:
                endCheckTimeStep = startCheckTimestep + minLength

                if endCheckTimeStep <= maxCheckingTimeStep:

                    pathParameters.update({'timeStep': (startCheckTimestep, endCheckTimeStep)})
                    trajectories = [trajectory[startCheckTimestep:endCheckTimeStep] for trajectory in leagelOriginalTrajctories]
                    stateIndex = 0
                    qPosIndex = [0, 1]
                    calculateChasingDeviation = CalculateChasingDeviation(sheepId, wolfId, stateIndex, qPosIndex)
                    trajectoryDeviationes = np.array([np.mean(calculateChasingDeviation(trajectory)) for trajectory in trajectories])
                    trajectoryLengthes = np.array([len(trajectory) for trajectory in trajectories])
                    calculateDistractorMoveDistance = CalculateDistractorMoveDistance(distractorId, stateIndex, qPosIndex)
                    calculateSheepMoveDistance = CalculateDistractorMoveDistance(sheepId, stateIndex, qPosIndex)
                    trajectoryDistractorMoveDistances = np.array([calculateDistractorMoveDistance(trajectory) for trajectory in trajectories])
                    trajectorySheepMoveDistances = np.array([np.mean(calculateSheepMoveDistance(trajectory)) for trajectory in trajectories])
                    countCross = CountSheepCrossRope(sheepId, wolfId, masterId, stateIndex, qPosIndex,tranformCoordinates, isCrossAxis)
                    try:
                        trajectoryCountCross =  np.array([countCross(trajectory) for trajectory in trajectories])
                    except:
                        print('ggg', pathParameters)
                        #print('ggg', trajectories)
                    collisionRadius = 1
                    countCollision = CountCollision(sheepId,wolfId,stateIndex, qPosIndex,collisionRadius,isCollision)
                    trajectoryCountCollision =  np.array([countCollision(trajectory) for trajectory in trajectories])

                    # print(len(trajectories))
                    # print(np.max(trajectoryLengthes))
                    minDeviation = math.pi/4
                    maxDeviation = math.pi/2.5

                    minDistractorMoveDistance = 0
                    maxDistractorMoveDistance = 100

                    deviationLegelTraj = filter(lambda x: x <= maxDeviation and x >= minDeviation, trajectoryDeviationes)
                    deviationLegelTrajIndex = [list(trajectoryDeviationes).index(i) for i in deviationLegelTraj]

                    timeWindow = 10
                    angleVariance = math.pi / 10
                    circleFilter = FindCirlceBetweenWolfAndMaster(wolfId, masterId, stateIndex, qPosIndex, timeWindow, angleVariance)
                    filterlist = [circleFilter(trajectory) for trajectory in trajectories]
                    timewindowLeagelTrajIndex = np.where(filterlist)[0]

                    lengthLeagelTrajIndex = np.nonzero(trajectoryLengthes >= minLength)
                    distractorMoveDistanceLegelTraj = filter(lambda x: x <= maxDistractorMoveDistance and x >= minDistractorMoveDistance, trajectoryDistractorMoveDistances)
                    distractorMoveDistanceLegelTrajIndex = [list(trajectoryDistractorMoveDistances).index(i) for i in distractorMoveDistanceLegelTraj]

                    acrossLegelTrajIndex = np.nonzero(trajectoryCountCross == 0 )
                    collisionLegelTrajIndex = np.nonzero(trajectoryCountCollision == 0)

                    leagelTrajIndex = ft.reduce(np.intersect1d, [deviationLegelTrajIndex, lengthLeagelTrajIndex, distractorMoveDistanceLegelTrajIndex, timewindowLeagelTrajIndex, acrossLegelTrajIndex, collisionLegelTrajIndex])
                    # leagelTrajIndex = list(range(len(trajectoryLengthes)))
                    # leagelTrajIndex = lengthLeagelTrajIndex[0]
                    if len(leagelTrajIndex) > 0:
                        print(leagelTrajIndex)
                        print('deviation:', trajectoryDeviationes[leagelTrajIndex])
                        print('mean deviation:', np.mean(trajectoryDeviationes[leagelTrajIndex]))

                        leagelTrajectories = [trajectory[:minLength] for trajectory in np.array(trajectories)[leagelTrajIndex]]
                        ropePartIndexes = list(range(3, 12))
                        offsetMasterStates = OffsetMasterStates(wolfId, masterId, ropePartIndexes, qPosIndex, stateIndex, masterDelayStep)
                        masterDelayedStates = [offsetMasterStates(trajectory) for trajectory in leagelTrajectories]
                        masterDelayedStatesPathParameters = pathParameters.copy()
                        masterDelayedStatesPathParameters['offset'] = masterDelayStep
                        masterDelayedStatesPath = getTrajectorySavePath(masterDelayedStatesPathParameters)
                        saveToPickle(masterDelayedStates, masterDelayedStatesPath)

                        demoStepIndex = list(range(masterDelayStep, minLength))
                        demoTajectories = [np.array(trajectory)[demoStepIndex] for trajectory in leagelTrajectories]
                        demoStates = np.array([[timestep[stateIndex] for timestep in trajectory] for trajectory in demoTajectories])
                        demoStatesPathParameters = pathParameters.copy()
                        demoStatesPathParameters['offset'] = 0
                        demoStatesPath = getTrajectorySavePath(demoStatesPathParameters)
                        saveToPickle(demoStates, demoStatesPath)
                        endTime = time.time()

                        print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
