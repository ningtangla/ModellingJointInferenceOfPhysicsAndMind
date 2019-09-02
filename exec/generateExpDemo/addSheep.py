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
import copy

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
from src.constrainedChasingEscapingEnv.demoFilter import OffsetMasterStates, TransposeRopePoses, ReplaceSheep, AddSheep, FindCirlceBetweenWolfAndMaster
from exec.generateExpDemo.filterTraj import CalculateChasingDeviation, CalculateDistractorMoveDistance, CountSheepCrossRope, isCrossAxis, tranformCoordinates, CountCollision, isCollision, IsAllInAngelRange,  CountCirclesBetweenWolfAndMaster


def main():
    startTime = time.time()

    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'generateExpDemo','trajectories')

    trajectoryExtension = '.pickle'

    numAgents = 3
    pureMCTSAgentId = 310
    sheepId = 0
    wolfId = 1
    masterId = 2
    distractorId = 3
    numSimulations = 200
    killzoneRadius = 2

    manipulatedVariables = OrderedDict()
    manipulatedVariables['selfIteration'] = [6000]
    manipulatedVariables['otherIteration'] = [6000]
    manipulatedVariables['selfId'] = [0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditionParametersAll = [dict(list(i)) for i in productedValues]

    trajectoryFixedParameters = {'killzoneRadius': killzoneRadius, 'numSimulations': numSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex', 'maxRunningSteps']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    stateIndex = 0
    addSheep = AddSheep(sheepId, stateIndex)
    for pathParameters in conditionParametersAll:
        trajectories = loadTrajectories(pathParameters)
        addSheepTrajs = addSheep(trajectories)
        addSheepTrajsPathParameters = copy.deepcopy(pathParameters)
        addSheepTrajsPathParameters['addSheep'] = 1
        addSheepTrajsPath = getTrajectorySavePath(addSheepTrajsPathParameters)
        saveToPickle(addSheepTrajs, addSheepTrajsPath)


if __name__ == '__main__':
    main()
