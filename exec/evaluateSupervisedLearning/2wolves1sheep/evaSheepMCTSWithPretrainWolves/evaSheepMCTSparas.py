import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, RewardFunctionWithWall
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


def main():
    numSimulationsList = [110, 150, 200]
    maxRolloutStepsList = [10, 15, 20]

    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'evaSheepMCTSWithPretrainWolves', 'trajectories')

    for numSimulations, maxRolloutSteps in it.product(numSimulationsList, maxRolloutStepsList):

        maxRunningSteps = 50
        killzoneRadius = 50
        sheepId = 0

        dataSetFixedParameters = {'agentId': sheepId, 'numSimulations': numSimulations, 'maxRolloutSteps': maxRolloutSteps}

        dataSetExtension = '.pickle'
        getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
        print("DATASET LOADED!")

        # accumulate rewards for trajectories
        numOfAgent = 3
        sheepId = 0
        wolf1Id = 1
        wolf2Id = 2
        xPosIndex = [0, 1]

        getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolf1Pos = GetAgentPosFromState(wolf1Id, xPosIndex)
        getWolf2Pos = GetAgentPosFromState(wolf2Id, xPosIndex)

        playAliveBonus = 1 / maxRunningSteps
        playDeathPenalty = -1
        playKillzoneRadius = killzoneRadius

        playIsTerminalByWolf1 = IsTerminal(playKillzoneRadius, getSheepPos, getWolf1Pos)
        playIsTerminalByWolf2 = IsTerminal(playKillzoneRadius, getSheepPos, getWolf2Pos)

        def playIsTerminal(state): return playIsTerminalByWolf1(state) or playIsTerminalByWolf2(state)

        playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

        decay = 1
        accumulateRewards = AccumulateRewards(decay, playReward)
        addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

        # pre-process the trajectories
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        numActionSpace = len(sheepActionSpace)
        actionToOneHot = ActionToOneHot(sheepActionSpace)

        actionIndex = 1

        def getTerminalActionFromTrajectory(trajectory): return trajectory[-1][actionIndex]
        removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
        processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
        preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

        fuzzySearchParameterNames = ['sampleIndex', 'timeUsed']
        loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
        loadedTrajectories = loadTrajectories(parameters={})
        # print(len(loadedTrajectories))

        def filterState(timeStep): return (timeStep[0][0:numOfAgent], timeStep[1], timeStep[2])
        trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]

        valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
        trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
        print(dataSetFixedParameters, trainDataMeanAccumulatedReward)


if __name__ == '__main__':
    main()
