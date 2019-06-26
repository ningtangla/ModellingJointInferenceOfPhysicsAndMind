import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNet import GenerateModelSeparateLastLayer, Train, saveVariables
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.neuralNetwork.trainTools import CoefficientController, TrainTerminalController, TrainReporter

import random
import numpy as np
import pickle
import functools as ft


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()

    return dataSet


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index
                                   in range(len(self.actionSpace))])

        return oneHotAction


class AccumulateRewards:
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array(
            [ft.reduce(accumulateReward, reversed(rewards[TimeT:])) for TimeT in range(len(rewards))])

        return accumulatedRewards


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionValueTriples = [triple for trajectory in trajectories for triple in trajectory]
        triplesFiltered = list(filter(lambda triple: triple[self.actionIndex] is not None, stateActionValueTriples))    # should I remove this None condition? because it will remove the terminal points--so we don't get Value prediction for those points.
        print("{} data points remain after filtering".format(len(triplesFiltered)))
        triplesProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
                                     for state, actions, value in triplesFiltered]

        return triplesProcessed


def main():
    # Get dataset for training
    dataSetDirectory = "../../data/testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco/trajectories"
    # dataSetDirectory = "../../data/testMCTSUniformVsNNPriorSheepChaseWolfMujoco/trajectories"
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 10
    dataSetNumSimulations = 75
    dataSetNumTrials = 1500
    dataSetQPosInit = (0, 0, 0, 0)
    dataSetSheepPolicyName = 'MCTS'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'sheepPolicyName': dataSetSheepPolicyName}
    dataSetPath = getDataSetPath(dataSetConditionVariables)

    dataSetTrajectories = loadData(dataSetPath)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)

    allValues = [accumulateRewards(trajectory) for trajectory in dataSetTrajectories]

    # combine trajectories with the accumulated reward for each
    trajectoriesWithValues = [[pair + (value,) for pair, value in zip(trajectory, trajectoryValues)]                    # is there a better way to do this? Maybe we should remove the lists altogether.
                                for trajectory, trajectoryValues in zip(dataSetTrajectories, allValues)]

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
    stateActionValueTriplesProcessed = preProcessTrajectories(trajectoriesWithValues)

    # shuffle and separate states and actions
    random.shuffle(stateActionValueTriplesProcessed)
    trainData = [[state for state, action, value in stateActionValueTriplesProcessed],
                 [action for state, action, value in stateActionValueTriplesProcessed],
                 np.asarray([[value for state, action, value in stateActionValueTriplesProcessed]]).T]

    # initialise model for training
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    # train models
    allTrainSteps = [0, 50000]
    batchSize = None
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientController(initActionCoeff, initValueCoeff)
    reportInterval = 100

    getTrain = lambda trainSteps: Train(trainSteps, batchSize, terminalController, coefficientController,
                                        TrainReporter(trainSteps, reportInterval))

    allTrainFunctions = {trainSteps: getTrain(trainSteps) for trainSteps in allTrainSteps}
    allTrainedModels = {trainSteps: train(generatePolicyNet(hiddenWidths), trainData) for trainSteps, train in
                        allTrainFunctions.items()}

    # get path to save trained models
    fixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                       'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                       'learnRate': learningRate, 'output': 'policyValue'}
    modelSaveDirectory = "../../data/testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, fixedParameters)
    allModelSavePaths = {trainedModel: getModelSavePath({'trainSteps': trainSteps}) for trainSteps, trainedModel in
                         allTrainedModels.items()}

    # save trained model variables
    savedVariables = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                      allModelSavePaths.items()]


if __name__ == '__main__':
    main()