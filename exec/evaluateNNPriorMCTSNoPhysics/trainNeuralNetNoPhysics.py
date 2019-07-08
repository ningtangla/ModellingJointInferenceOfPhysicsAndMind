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
    dataSetNumTrials = 5000
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

    # train models
    allTrainSteps = [0, 100, 500]
    reportInterval = 100
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    getTrain = lambda trainSteps: Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                                        summaryOn=False, testData=None)

    allTrainFunctions = {trainSteps: getTrain(trainSteps) for trainSteps in allTrainSteps}
    allTrainedModels = {trainSteps: train(generatePolicyNet(hiddenWidths), trainData) for trainSteps, train in
                        allTrainFunctions.items()}

    # get path to save trained models
    fixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps,
                       'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                       'learnRate': learningRate}
    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, fixedParameters)
    allModelSavePaths = {trainedModel: getModelSavePath({'trainSteps': trainSteps}) for trainSteps, trainedModel in
                         allTrainedModels.items()}

    # save trained model variables
    savedVariables = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                      allModelSavePaths.items()]