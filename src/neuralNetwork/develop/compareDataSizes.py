import numpy as np
import random
import pickle
import policyValueNet as net
import dataTools
import sheepEscapingEnv as env
import visualize as VI
import trainTools


def main(seed=128, tfseed=128):
	random.seed(seed)
	np.random.seed(4027)

	dataSetPath = "72640steps_1000trajs_sheepEscapingEnv_data_actionDist.pkl"
	dataSet = dataTools.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [1000]  # [5000, 15000, 30000, 45000, 60000]
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:size])] for size in trainingDataSizes]

	testDataSize = 12640
	testData = [list(varData) for varData in zip(*dataSet[-testDataSize:])]

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	learningRate = 1e-4
	regularizationFactor = 0
	valueRelativeErrBound = 0.1
	generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
	models = [generateModel([64, 64, 64, 64]) for _ in range(len(trainingDataSizes))]

	net.restoreVariables(models[0], "savedModels/60000data_64x4_minibatch_100kIter_contState_actionDist")

	maxStepNum = 100000
	batchSize = None
	reportInterval = 1000
	lossChangeThreshold = 1e-8
	lossHistorySize = 10
	initActionCoefficient = 50
	initValueCoefficient = 1
	trainTerminalController = trainTools.TrainTerminalController(lossHistorySize, lossChangeThreshold)
	coefficientController = trainTools.coefficientCotroller(initActionCoefficient, initValueCoefficient)
	trainReporter = trainTools.TrainReporter(maxStepNum, reportInterval)
	train = net.Train(maxStepNum, batchSize, trainTerminalController, coefficientController, trainReporter)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

	evalTrain = {("Train", size): list(net.evaluate(model, trainingData).values()) for size, trainingData, model in
	             zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTest = {("Test", size): list(net.evaluate(model, testData).values()) for size, trainingData, model in
	            zip(trainingDataSizes, trainingDataList, trainedModels)}
	evalTrain.update(evalTest)

	print(evalTrain)
	# saveFile = open("diffDataSizesModels/{}evalResults.pkl".format(trainingDataSizes), "wb")
	# pickle.dump(evalTrain, saveFile)

	# VI.draw(evalTrain, ["mode", "training_set_size"], ["actionLoss", "actionAcc", "valueLoss", "valueAcc"])

	# for size, model in zip(trainingDataSizes, trainedModels):
	# 	net.saveVariables(model, "diffDataSizesModels/{}data_64x4_minibatch_{}kIter_contState_actionDist".format(size, int(maxStepNum/1000)))
	# net.saveVariables(trainedModels[0], "savedModels/iter_60000data_64x4_minibatch_200kIter_contState_actionDist")


if __name__ == "__main__":
	main(20)
