import sys
sys.path.append('../../src/neuralNetwork/')
sys.path.append('..')

from evaluationFunctions import GetSavePath
from policyNet import GenerateModel, Train, saveVariables

import random
import os
import numpy as np
import pickle


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()
    return dataSet


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = [1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index in
                        range(len(self.actionSpace))]

        return oneHotAction


class PreprocessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [pair for trajectory in dataSetTrajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(lambda pair: pair[self.actionIndex] is not None, stateActionPairs))
        stateActionPairsProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(action[self.agentId]))
                                     for state, action in stateActionPairsFiltered]

        return stateActionPairsProcessed


if __name__ == '__main__':
    # Get dataset for training
    dataSetDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 5#15
    dataSetNumSimulations = 5#2
    dataSetNumTrials = 7 #1
    dataSetQPosInit = (-4, 0, 4, 0)
    dataSetSheepPolicyName = 'mcts'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'sheepPolicyName': dataSetSheepPolicyName}
    dataSetPath = getDataSetPath(dataSetConditionVariables)

    dataSetTrajectories = loadData(dataSetPath)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionToOneHot = ActionToOneHot(actionSpace)
    sheepId = 0
    actionIndex = 1
    preprocessTrajectories = PreprocessTrajectories(sheepId, actionIndex, actionToOneHot)
    stateActionPairsProcessed = preprocessTrajectories(dataSetTrajectories)

    # shuffle and separate states and actions
    random.shuffle(stateActionPairsProcessed)
    trainData = [[state for state, action in stateActionPairsProcessed],
                 [action for state, action in stateActionPairsProcessed]]

    # initialise model for training
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    modelToTrain = generatePolicyNet(hiddenWidths)

    # train model
    trainSteps = 50000
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    train = Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                     summaryOn=False, testData=None)

    trainedModel = train(modelToTrain, trainData)

    # get path to save trained model
    modelTrainConditions = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                            'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                            'learnRate': learningRate, 'trainSteps': trainSteps}
    modelSaveDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)
    modelSavePath = getModelSavePath(modelTrainConditions)

    # save trained model variables
    saveVariables(trainedModel, modelSavePath)
