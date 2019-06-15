import numpy as np
import tensorflow as tf
import random
import supervisedLearning as SL
import neuralNetwork as NN
import prepareNeuralNetData as PD


if __name__ == "__main__":
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    numStateSpace = 12
    numActionSpace = 8
    dataSetPath = "sheepWolfMujocoData.pkl"
    dataSet = PD.loadData(dataSetPath)
    random.shuffle(dataSet)

    trainingDataSizes = [3000]  # list(range(10000, 10000, 1000))
    trainingDataList = [([state for state, _ in dataSet[:size]], [label for _, label in dataSet[:size]]) for size in
                        trainingDataSizes]
    testDataSize = 5000
    testData = PD.sampleData(dataSet, testDataSize)

    learningRate = 0.0001
    regularizationFactor = 1e-4
    generatePolicyNet = NN.GeneratePolicyNet(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    models = [generatePolicyNet(2, 128) for _ in range(len(trainingDataSizes))]

    maxStepNum = 50000
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    train = SL.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                     summaryOn=False, testData=None)

    trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

    evalTrain = {("Train", len(trainingData[0])): SL.evaluate(model, trainingData) for trainingData, model in
                 zip(trainingDataList, trainedModels)}
    evalTest = {("Test", len(trainingData[0])): SL.evaluate(model, testData) for trainingData, model in
                zip(trainingDataList, trainedModels)}
    evalTrain.update(evalTest)

    print(evalTrain)
