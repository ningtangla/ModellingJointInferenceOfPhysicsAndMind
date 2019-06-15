import policyValueNet as net
import sheepEscapingEnv as env
import numpy as np
import pandas as pd
from anytree import AnyNode as Node
from AnalyticGeometryFunctions import calculateCrossEntropy
from pylab import plt
import pickle
import itertools as it
import mcts

def getNNModel(modelPath):
	generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0,
	                                                   regularizationFactor=0, valueRelativeErrBound=0.0)
	model = generateModel([64, 64, 64, 64])
	trainedModel = net.restoreVariables(model, modelPath)
	return trainedModel


def getMCTSModel():
	xBoundary = env.xBoundary
	yBoundary = env.yBoundary
	actionSpace = env.actionSpace

	wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
	transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
	isTerminal = env.IsTerminal(minDistance=env.vel + 5)

	rewardFunction = lambda state, action: 1

	cInit = 1
	cBase = 1
	calculateScore = mcts.CalculateScore(cInit, cBase)
	selectChild = mcts.SelectChild(calculateScore)

	getActionPrior = mcts.UniformActionPrior(actionSpace)
	initializeChildren = mcts.InitializeChildren(actionSpace, transition, getActionPrior)
	expand = mcts.Expand(transition, isTerminal, initializeChildren)

	maxRollOutSteps = 5
	rolloutPolicy = lambda state: actionSpace[np.random.choice(range(env.numActionSpace))]
	nodeValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)

	numSimulations = 100
	getPolicyOutput = mcts.getSoftmaxActionDist
	policy = mcts.MCTSPolicy(numSimulations, selectChild, expand, nodeValue, mcts.backup, getPolicyOutput)
	return policy


def getNodeFromState(state):
	rootNode = Node(id={None: state}, num_visited=0, sum_value=0, is_expanded=False)
	return rootNode


def evaluateModel(df, mctsPolicy, nnModel, sampleFunction):
	validStates = sampleFunction(df.index[0])
	if len(validStates) == 0:
		return pd.Series({"cross_entropy": 0})
	# validNodes = [getNodeFromState(state) for state in validStates]
	actionDistribution = [mctsPolicy(state) for state in validStates]
	nnActionDistribution = [net.approximateActionPrior(state, nnModel, env.actionSpace) for state in validStates]
	crossEntropyList = [calculateCrossEntropy(np.array(list(prediction.values())), np.array(list(target.values()))) for prediction, target in zip(nnActionDistribution, actionDistribution)]
	meanCrossEntropy = np.mean(crossEntropyList)
	# print(actionDistribution)
	# print(nnActionDistribution)
	return pd.Series({"cross_entropy": meanCrossEntropy})


def drawHeatMap(dataDF, groupByVariableNames, subplotIndex, subplotIndexName, valueIndexName):
	figure = plt.figure(figsize=(12, 10))
	order = [7,8,9,4,5,6,1,2,3]
	numOfplot = 0
	for key, subDF in dataDF.groupby(groupByVariableNames):
		subplot = figure.add_subplot(subplotIndex[0], subplotIndex[1], order[numOfplot])
		plotDF = subDF.reset_index()
		plotDF.plot.scatter(x=subplotIndexName[0], y=subplotIndexName[1], c=valueIndexName, colormap="jet", ax=subplot, vmin=0, vmax=9)
		numOfplot = numOfplot + 1
	plt.subplots_adjust(wspace=0.8, hspace=0.4)
	plt.savefig("./Graphs/Factor3_fixed_50sample_actionDistribution_divergence_HeatMap.png")


class SampleStatesByIndex():
	def __init__(self, terminal, num, discreteRange, wolfBias, sheepBias):
		self.terminal = terminal
		self.num = num
		self.discreteRange = discreteRange
		self.xLowerBound, self.xUpperBound = env.xBoundary
		self.yLowerBound, self.yUpperBound = env.yBoundary
		self.wolfBias = wolfBias
		self.sheepBias = sheepBias

	def __call__(self, index):
		wolfUpperBound, sheepUpperBound = index[0:2], index[2:4]
		if wolfUpperBound[0] == self.xUpperBound:
			wolfX = wolfUpperBound[0] - self.wolfBias
		elif wolfUpperBound[0] - self.discreteRange == self.xLowerBound:
			wolfX = self.xLowerBound + self.wolfBias
		else:
			wolfX = self.xLowerBound + 2*self.wolfBias
		if wolfUpperBound[1] == self.yUpperBound:
			wolfY = wolfUpperBound[1] - self.wolfBias
		elif wolfUpperBound[1] - self.discreteRange == self.yLowerBound:
			wolfY = self.yLowerBound + self.wolfBias
		else:
			wolfY = self.yLowerBound + 2*self.wolfBias
		wolfPosition = np.array([wolfX, wolfY])

		if sheepUpperBound[0] == self.xUpperBound:
			sheepX = sheepUpperBound[0] - self.sheepBias
		elif sheepUpperBound[0] - self.discreteRange == self.xLowerBound:
			sheepX = self.xLowerBound + self.sheepBias
		else:
			sheepX = self.xLowerBound + 2 * self.wolfBias
		if sheepUpperBound[1] == self.yUpperBound:
			sheepY = sheepUpperBound[1] - self.sheepBias
		elif sheepUpperBound[1] - self.discreteRange == self.yLowerBound:
			sheepY = self.yLowerBound + self.sheepBias
		else:
			sheepY = self.yLowerBound + 2 * self.wolfBias
		sheepPosition = np.array([sheepX, sheepY])

		state = np.concatenate([wolfPosition, sheepPosition])
		states = [state for count in range(self.num) if not self.terminal(state)]
		return states


class FixedPtsSample:
	def __init__(self, allStates, terminal, discreteRange, num):
		self.allStates = allStates
		self.discreteRange = discreteRange
		self.num = num
		self.terminal = terminal

	def __call__(self, index):
		for state in self.allStates:
			valid = True
			for cnt in range(len(state)):
				if state[cnt] > index[cnt] or state[cnt] <= index[cnt] - self.discreteRange:
					valid = False
					break
			if valid:
				return [state for count in range(self.num) if not self.terminal(state)]


def saveMapData(df, path):
	file = open(path, "wb")
	pickle.dump(df, file)
	file.close()

def loadMapData(path):
	file = open(path, "rb")
	df = pickle.load(file)
	file.close()
	return df

def main():
	discreteFactor = 3
	savePath = "./FixedPts_HeatMapData.pkl"
	save = False
	mindistance = env.vel + 5
	print("discreteFactor:{}".format(discreteFactor))
	discreteRange = env.xBoundary[1] / discreteFactor
	numOfPoints = 50
	wolfXPosition = [env.xBoundary[1]/discreteFactor * (i+1) for i in range(discreteFactor)]
	wolfYPosition = [env.yBoundary[1]/discreteFactor * (i+1) for i in range(discreteFactor)]
	sheepXPosition = wolfXPosition
	sheepYPosition = wolfYPosition
	levelValues = [wolfXPosition, wolfYPosition, sheepXPosition, sheepYPosition]
	levelNames = ["wolfXPosition", "wolfYPosition", "sheepXPosition", "sheepYPosition"]
	diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
	toSplitFrame = pd.DataFrame(index=diffIndex)

	modelPath = "savedModels/60000data_64x4_minibatch_100kIter_contState_actionDist"
	nnModel = getNNModel(modelPath)
	mctsModel = getMCTSModel()
	isTerminal = env.IsTerminal(minDistance=mindistance)

	wolfFixedX = [45,90,135]
	wolfFixedY = [45,90,135]
	sheepFixedX = [15,90,165]
	sheepFixedY = [15,90,165]
	allStates = it.product(wolfFixedX, wolfFixedY, sheepFixedX, sheepFixedY)
	fixedPtsSampleFunction = FixedPtsSample(list(allStates), isTerminal, discreteRange, numOfPoints)
	fixedDF = toSplitFrame.groupby(levelNames).apply(evaluateModel, mctsModel, nnModel, fixedPtsSampleFunction)
	print(fixedDF)
	exit(0)
	if save:
		saveMapData(fixedDF, savePath)
	drawHeatMap(fixedDF, ['wolfYPosition', 'wolfXPosition'], [len(wolfXPosition), len(wolfYPosition)], ['sheepXPosition', 'sheepYPosition'], 'cross_entropy')


if __name__ == "__main__":
	np.random.seed(5)
	main()