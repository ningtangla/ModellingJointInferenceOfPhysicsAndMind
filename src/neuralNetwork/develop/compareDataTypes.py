import numpy as np
import random
import pickle
import policyValueNet as net
import dataTools
import visualize as VI
import sheepEscapingEnv as env


def main(seed=128, tfseed=128):
	random.seed(seed)
	np.random.seed(seed)

	control_path = "35087steps_500trajs_sheepEscapingEnv_data_actionDist.pkl"
	actionLabel_path = "35087steps_500trajs_sheepEscapingEnv_data_actionLabel.pkl"
	discState_path = "35087steps_500trajs_sheepEscapingEnv_data_discState_actionDist.pkl"
	nearWallInit_path = "10222steps_500trajs_sheepEscapingEnv_actionDist_initNearWall.pkl"
	controlSet = dataTools.loadData(control_path)
	actionLabelSet = dataTools.loadData(actionLabel_path)
	discStateSet = dataTools.loadData(discState_path)
	nearWallInitSet = dataTools.loadData(nearWallInit_path)
	nearWallBiasedSet = controlSet[:20000] + nearWallInitSet[:10000]

	trainingDataTypes = {"control": controlSet, "actionLabel": actionLabelSet, "discState": discStateSet, "nearWallBiased": nearWallBiasedSet}
	trainingDataSize = 30000
	trainingDataList = [[list(varData) for varData in zip(*dataSet[:trainingDataSize])] for dataSet in trainingDataTypes.values()]

	numStateSpace = env.numStateSpace
	numActionSpace = env.numActionSpace
	learningRate = 1e-4
	regularizationFactor = 0
	valueRelativeErrBound = 0.1
	generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor, valueRelativeErrBound=valueRelativeErrBound, seed=tfseed)
	models = [generateModel([64, 64, 64, 64]) for _ in range(len(trainingDataTypes))]

	# net.restoreVariables(models[0], "./savedModels/64*4_30000steps_33%nearWall_minibatch_contState_actionDist")

	maxStepNum = 100000
	batchSize = 4096
	reportInterval = 1000
	lossChangeThreshold = 1e-8
	lossHistorySize = 10
	train = net.Train(maxStepNum, batchSize, lossChangeThreshold, lossHistorySize, reportInterval,
	                  summaryOn=False, testData=None)

	trainedModels = [train(model, data) for model, data in zip(models, trainingDataList)]

	evalTrain = {("Train", size): list(net.evaluate(model, trainingData).values()) for size, trainingData, model in
	             zip(trainingDataTypes.keys(), trainingDataList, trainedModels)}

	print(evalTrain)
	saveFile = open("diffDataTypesModels/[{}]evalResults.pkl".format(','.join(list(trainingDataTypes.keys()))), "wb")
	pickle.dump(evalTrain, saveFile)

	# VI.draw(evalTrain, ["mode", "training_set_size"], ["actionLoss", "actionAcc", "valueLoss", "valueAcc"])

	for type, model in zip(trainingDataTypes.keys(), trainedModels):
		net.saveVariables(model, "diffDataTypesModels/{}_30000data_64x4_minibatch_{}kIter".format(type, int(maxStepNum/1000)))


if __name__ == "__main__":
	main()
