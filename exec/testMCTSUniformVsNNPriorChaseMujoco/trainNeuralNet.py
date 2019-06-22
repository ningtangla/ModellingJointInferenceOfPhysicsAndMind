import sys
import os
sys.path.append(os.path.join('..', '..', 'src', 'neuralNetwork'))
sys.path.append(os.path.join('..', '..', 'src', 'sheepWolf'))
sys.path.append('..')

from evaluationFunctions import GetSavePath
from policyNet import GenerateModel, Train, saveVariables
from sheepWolfWrapperFunctions import GetAgentPosFromState

import random
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


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [pair for trajectory in trajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(lambda pair: pair[self.actionIndex] is not None and pair[0][1][2] < 9.7, stateActionPairs))
        print("{} data points remain after filtering".format(len(stateActionPairsFiltered)))
        stateActionPairsProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]))
                                     for state, actions in stateActionPairsFiltered]

        return stateActionPairsProcessed


if __name__ == '__main__':
    # Get dataset for training
    dataSetDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
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

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionToOneHot = ActionToOneHot(actionSpace)
    sheepId = 0
    actionIndex = 1
    stateIndex = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
    stateActionPairsProcessed = preProcessTrajectories(dataSetTrajectories)

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

    # train models
    allTrainSteps = [20]#[0, 50, 100, 500]#[1000, 5000, 10000, 15000]
    reportInterval = 100
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    getTrain = lambda trainSteps: Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                                        summaryOn=False, testData=None)

    allTrainFunctions = {trainSteps: getTrain(trainSteps) for trainSteps in allTrainSteps}
    allTrainedModels = {trainSteps: train(generatePolicyNet(hiddenWidths), trainData) for trainSteps, train in
                        allTrainFunctions.items()}

    # get path to save trained models
    fixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                       'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                       'learnRate': learningRate}
    modelSaveDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, fixedParameters)
    allModelSavePaths = {trainedModel: getModelSavePath({'trainSteps': trainSteps}) for trainSteps, trainedModel in
                         allTrainedModels.items()}

    # save trained model variables
    savedVariables = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                      allModelSavePaths.items()]