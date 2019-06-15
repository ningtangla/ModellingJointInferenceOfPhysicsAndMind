import numpy as np
import tensorflow as tf
import itertools as it
import random
import pickle
import supervisedLearning as SL
import neuralNetwork as NN
import prepareDataContinuousEnv as PD
import visualize as VI


if __name__ == "__main__":
	random.seed(128)
	np.random.seed(128)
	tf.set_random_seed(128)

	numStateSpace = 4
	numActionSpace = 8
	dataSetPath = 'continuous_data.pkl'
	dataSet = PD.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [8000] #list(range(10000, 10000, 1000))
	trainingDataList = [([state for state, _ in dataSet[:size]], [label for _, label in dataSet[:size]]) for size in trainingDataSizes]
	trainingData = trainingDataList[0]
	learningRate = 0.0001
	regularizationFactor = 1e-4
	generatePolicyNet = NN.GeneratePolicyNet(numStateSpace, numActionSpace, learningRate, regularizationFactor)
	policyNetDepth = [2, 3, 4, 5]
	policyNetWidth = [32, 64, 128, 256]
	nnList = it.product(policyNetDepth, policyNetWidth)
	models = {(width, depth): generatePolicyNet(depth, width) for depth, width in nnList}

	maxStepNum = 50000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = SL.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
	                 summaryOn=False, testData=None)

	trainedModels = {key: train(model, trainingData) for key, model in models.items()}

	evalTrain = {key: SL.evaluate(model, trainingData) for key, model in trainedModels.items()}

	evaluateDataPath = 'NeuralNetworkEvaluation.pkl'
	file = open(evaluateDataPath, "wb")
	pickle.dump(dataSet, file)
	file.close()
	VI.draw(evalTrain, ["neurons per layer", "layer"], ["Loss", "Accuracy"])