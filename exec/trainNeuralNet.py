import sys
sys.path.append('../src/neuralNetwork/')

from evaluationFunctions import GetSavePath
from prepareNeuralNetData import loadData
from neuralNetwork import GeneratePolicyNet
from mainNeuralNet import flattenStates
from supervisedLearning import Train

import random
import pickle


if __name__ == '__main__':
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numStateSpace = 12
    numActionSpace = len(actionSpace)

    dataSetDirectory = "../data/testNNPriorMCTSMujoco/trajectories"
    extension = 'pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, extension)
    maxRunningSteps = 15
    qPosInit = (-4, 0, 4, 0)
    numSimulations = 200
    numTrials = 1
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit, 'numSimulations': numSimulations,
                          'numTrials': numTrials}
    dataSetPath = getDataSetPath(conditionVariables)

    trajectories = loadData(dataSetPath)
    random.shuffle(trajectories)
    trainingData = ([state for state, action in trajectories], [action for state, action in trajectories])
    trainingDataFlat = flattenStates(trainingData)

    learningRate = 0.0001
    regularizationFactor = 1e-4
    generatePolicyNet = GeneratePolicyNet(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    modelToTrain = generatePolicyNet(2, 128)

    maxStepNum = 50000
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    train = Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                     summaryOn=False, testData=None)

    trainedModel = train(modelToTrain, trainingDataFlat)