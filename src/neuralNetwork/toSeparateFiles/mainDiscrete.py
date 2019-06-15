import numpy as np
import itertools as it
import tensorflow as tf
import math
import random
import supervisedLearning as SL
import neuralNetwork as NN
import gridEnv as GE
import prepareData as PD
import visualize as VI


if __name__ == '__main__':
	random.seed(128)
	np.random.seed(128)
	tf.set_random_seed(128)

	gridSize = 10
	transitionFunction = GE.TransitionFunction(gridSize)
	isTerminal = GE.IsTerminal()

	actionSpace = [[0,1], [1,0], [-1,0], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]]
	agentStates = [state for state in it.product([x for x in range(gridSize)], [y for y in range(gridSize)])]
	targetStates = [state for state in it.product([x for x in range(gridSize)], [y for y in range(gridSize)])]
	stateSpace = [state for state in it.product(agentStates, targetStates)]

	reset = GE.Reset(actionSpace, agentStates, targetStates)
	print('Generating Optimal Policy...')
	optimalPolicy = PD.generateOptimalPolicy(stateSpace, actionSpace)
	print('Optimal Policy Generated.')

	maxTimeStep = int(gridSize * gridSize / 2)
	sampleTrajectory = PD.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

	trajNum = 5000
	dataSetPath = "all_data.pkl"
	#PD.generateData(sampleTrajectory, optimalPolicy, trajNum, dataSetPath, actionSpace)
	dataSet = PD.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = list(range(1000, 9900, 1000))
	trainingDataList = [([state for state, _ in dataSet[:size]], [label for _, label in dataSet[:size]]) for size in trainingDataSizes]
	testDataSize = 5000
	testData = PD.sampleData(dataSet, testDataSize)

	learningRate = 0.001
	regularizationFactor = 1e-4
	generatePolicyNet = NN.GeneratePolicyNet(4, 8, learningRate, regularizationFactor)
	models = [generatePolicyNet(3, 32) for _ in range(len(trainingDataSizes))]

	maxStepNum = 50000
	reportInterval = 500
	lossChangeThreshold = 1e-6
	lossHistorySize = 10
	train = SL.Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
					 summaryOn=False, testData=None)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]
	# evalResults = [(SL.evaluate(model, trainingData), SL.evaluate(model, testData)) for trainingData, model in zip(trainingDataList, trainedModels)]

	evalTrain = {("Train", len(trainingData[0])): SL.evaluate(model, trainingData) for trainingData, model in zip(trainingDataList, trainedModels)}
	evalTest = {("Test", len(trainingData[0])): SL.evaluate(model, testData) for trainingData, model in zip(trainingDataList, trainedModels)}
	evalTrain.update(evalTest)

	VI.draw(evalTrain, ["mode", "training_set_size"], ["Loss", "Accuracy"])