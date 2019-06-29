import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyNet import GenerateModel, Train, saveVariables

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


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [pair for trajectory in trajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(lambda pair: pair[self.actionIndex] is not None, stateActionPairs))
        stateActionPairsProcessed = [(np.asarray(state).flatten().tolist(), self.actionToOneHot(action[self.agentId]))
                                     for state, action in stateActionPairsFiltered]

        return stateActionPairsProcessed


if __name__ == '__main__':
    # Get dataset for training
    dataSetDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 30
    dataSetNumSimulations = 200
    dataSetNumTrials = 2
    dataSetSheepPolicyName = 'mcts'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'sheepPolicyName': dataSetSheepPolicyName}
    dataSetPath = getDataSetPath(dataSetConditionVariables)
    dataSetTrajectories = loadData(dataSetPath)

    # pre-process the trajectories
    sheepId = 0
    actionIndex = 1
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
    stateActionPairsProcessed = preProcessTrajectories(dataSetTrajectories)

    # shuffle and separate states and actions
    random.shuffle(stateActionPairsProcessed)
    trainData = [[state for state, action in stateActionPairsProcessed],
                 [action for state, action in stateActionPairsProcessed]]

    # initialise model for training
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    modelToTrain = generatePolicyNet(hiddenWidths)

    # train model
    trainSteps = 10000
    lossChangeThreshold = 1e-8
    lossHistorySize = 10
    reportInterval = 500

    train = Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                  summaryOn=False, testData=None)

    trainedModel = train(modelToTrain, trainData)

    # get path to save trained model
    modelTrainConditions = {'maxRunningSteps': dataSetMaxRunningSteps,
                            'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                            'learnRate': learningRate, 'trainSteps': trainSteps}
    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)
    modelSavePath = getModelSavePath(modelTrainConditions)

    # save trained model variables
    saveVariables(trainedModel, modelSavePath)
