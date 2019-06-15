import numpy as np
import pandas as pd
import random
import os
import policyValueNet as net
import dataTools
import sheepEscapingEnv as env
import trainTools
import visualize as VI


class ApplyFunction:
	def __init__(self, saveModelDir=None, saveGraphDir=None):
		self.saveModelDir = saveModelDir
		self.saveGraphDir = saveGraphDir

	def __call__(self, df, dataSet, criticFunction, tfseed):
		trainDataSize = df.index.get_level_values('trainingDataSize')[0]
		trainData =  [list(varData) for varData in zip(*dataSet[:trainDataSize])]
		testDataSize = df.index.get_level_values('testDataSize')[0]
		testData = [list(varData) for varData in zip(*dataSet[-testDataSize:])]
		numStateSpace = df.index.get_level_values('numStateSpace')[0]
		numActionSpace = df.index.get_level_values('numActionSpace')[0]
		learningRate = df.index.get_level_values('learningRate')[0]
		regularizationFactor = df.index.get_level_values('regularizationFactor')[0]
		valueRelativeErrBound = df.index.get_level_values('valueRelativeErrBound')[0]
		maxStepNum = df.index.get_level_values('maxStepNum')[0]
		batchSize = df.index.get_level_values('batchSize')[0]
		lossChangeThreshold = df.index.get_level_values('lossChangeThreshold')[0]
		lossHistorySize = df.index.get_level_values('lossHistorySize')[0]
		initActionCoefficient = df.index.get_level_values('initActionCoefficient')[0]
		initValueCoefficient = df.index.get_level_values('initValueCoefficient')[0]
		netNeurons = df.index.get_level_values('netNeurons')[0]
		netLayers = df.index.get_level_values('netLayers')[0]
		neuronsPerLayer = int(round(netNeurons/netLayers))
		reportInterval = df.index.get_level_values('reportInterval')[0]

		trainTerminalController = trainTools.TrainTerminalController(lossHistorySize, lossChangeThreshold)
		coefficientController = trainTools.coefficientCotroller(initActionCoefficient, initValueCoefficient)
		trainReporter = trainTools.TrainReporter(maxStepNum, reportInterval)
		train = net.Train(maxStepNum, batchSize, trainTerminalController, coefficientController, trainReporter)

		generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
		model = generateModel([neuronsPerLayer] * netLayers)
		trainedModel = train(model, trainData)
		modelName = "{}data_{}x{}_minibatch_{}kIter_contState_actionDist".format(trainData, neuronsPerLayer, netLayers,
		                                                                         maxStepNum / 1000)
		if self.saveModelDir is not None:
			savePath = os.path.join(os.getcwd(), self.saveModelDir, modelName)
			net.saveVariables(trainedModel, savePath)

		evalTest = net.evaluate(trainedModel, testData)
		return pd.Series({"testActionLoss": evalTest['actionLoss']})

def main(seed=128, tfseed=128):
	random.seed(seed)
	np.random.seed(4027)

	dataSetPath = "72640steps_1000trajs_sheepEscapingEnv_data_actionDist.pkl"
	dataSet = dataTools.loadData(dataSetPath)
	random.shuffle(dataSet)

	trainingDataSizes = [100,200]  # [5000, 15000, 30000, 45000, 60000]
	testDataSize = [100]
	numStateSpace = [env.numStateSpace]
	numActionSpace = [env.numActionSpace]
	learningRate = [1e-4]
	regularizationFactor = [0]
	valueRelativeErrBound = [0.1]
	maxStepNum = [100000]
	batchSize = [100]
	reportInterval = [1000]
	lossChangeThreshold = [1e-8]
	lossHistorySize = [10]
	initActionCoefficient = [50]
	initValueCoefficient = [1]
	netNeurons = [256]
	netLayers = [4]

	levelNames = ["trainingDataSize", "testDataSize", "numStateSpace", "numActionSpace", "learningRate",
	              "regularizationFactor", "valueRelativeErrBound", "maxStepNum", "batchSize", "reportInterval",
	              "lossChangeThreshold", "lossHistorySize", "initActionCoefficient", "initValueCoefficient",
	              "netNeurons", "netLayers"]
	levelValues = [trainingDataSizes, testDataSize, numStateSpace, numActionSpace, learningRate, regularizationFactor,
	               valueRelativeErrBound, maxStepNum, batchSize, reportInterval, lossChangeThreshold, lossHistorySize,
	               initActionCoefficient, initValueCoefficient, netNeurons, netLayers]
	levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
	toSplitFrame = pd.DataFrame(index=levelIndex)

	applyFunctoin = ApplyFunction()
	resultDF = toSplitFrame.groupby(levelNames).apply(applyFunctoin, dataSet, None, tfseed)
	print(resultDF)


if __name__ == "__main__":
	main(20)
