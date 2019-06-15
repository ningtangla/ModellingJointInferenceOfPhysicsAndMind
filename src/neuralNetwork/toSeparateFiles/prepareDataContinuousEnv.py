import numpy as np
import pickle
import random
from AnalyticGeometryFunctions import computeAngleBetweenVectors


class OptimalPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		targetState = state[2:4]
		agentState = state[0:2]
		relativeVector = np.array(targetState) - np.array(agentState)
		angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
							   np.array(self.actionSpace)}
		action = angleBetweenVectors[min(angleBetweenVectors.keys())]
		return action


class SampleTrajectory:
	def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
		self.maxTimeStep = maxTimeStep
		self.transitionFunction = transitionFunction
		self.isTerminal = isTerminal
		self.reset = reset

	def __call__(self, policy):
		state = self.reset()
		while self.isTerminal(state):
			state = self.reset()
		trajectory = []
		for _ in range(self.maxTimeStep):
			action = policy(state)
			trajectory.append((state, action))
			newState = self.transitionFunction(state, action)
			if self.isTerminal(newState):
				break
			state = newState
		return trajectory


def generateData(sampleTrajectory, policy, actionSpace, trajNumber, path):
	totalStateBatch = []
	totalActionBatch = []
	for index in range(trajNumber):
		if index % 100 == 0: print(index)
		trajectory = sampleTrajectory(policy)
		states, actions = zip(*trajectory)
		totalStateBatch = totalStateBatch + list(states)
		oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalActionBatch = totalActionBatch + oneHotActions
	dataSet = list(zip(totalStateBatch, totalActionBatch))
	saveFile = open(path, "wb")
	pickle.dump(dataSet, saveFile)


def loadData(path):
	pklFile = open(path, "rb")
	dataSet = pickle.load(pklFile)
	pklFile.close()
	return dataSet


def sampleData(data, batchSize):
	batch = random.sample(data, batchSize)
	batchInput = [x for x, _ in batch]
	batchOutput = [y for _, y in batch]
	return batchInput, batchOutput


def prepareDataContinuousEnv():
	actionSpace = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
	policy = OptimalPolicy(actionSpace)

	maxTimeStep = 180
	import continuousEnv
	xbound = [0, 180]
	ybound = [0, 180]
	vel = 1
	transitionFunction = continuousEnv.TransitionFunction(xbound, ybound, vel)
	isTerminal = continuousEnv.IsTerminal(1.5*vel)
	reset = continuousEnv.Reset(xbound, ybound)
	sampleTraj = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

	trajNum = 1200
	path = "./continuous_data.pkl"
	generateData(sampleTraj, policy, actionSpace, trajNum, path)

	data = loadData(path)
	print("{} data points in {}".format(data, path))


if __name__ == "__main__":
	prepareDataContinuousEnv()
