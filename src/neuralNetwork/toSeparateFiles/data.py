import numpy as np
import pickle
import random
import functools as ft


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


class AccumulateRewards():
	def __init__(self, decay, rewardFunction):
		self.decay = decay
		self.rewardFunction = rewardFunction
	def __call__(self, trajectory):
		rewards = [self.rewardFunction(state, action) for state, action in trajectory]
		accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
		accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
		return accumulatedRewards


def generateData(sampleTrajectory, accumulateRewards, policy, actionSpace, trajNumber, path, withReward=True):
	totalStateBatch = []
	totalActionBatch = []
	totalRewardBatch = []
	for index in range(trajNumber):
		if index % 100 == 0: print("{} trajectories generated".format(index))
		trajectory = sampleTrajectory(policy)
		if withReward:
			accumulatedRewards = accumulateRewards(trajectory)
			totalRewardBatch = totalRewardBatch + list(accumulatedRewards)
		states, actions = zip(*trajectory)
		totalStateBatch = totalStateBatch + list(states)
		oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in range(len(actionSpace))] for action in actions]
		totalActionBatch = totalActionBatch + oneHotActions
	if withReward:
		dataSet = list(zip(totalStateBatch, totalActionBatch, totalRewardBatch))
	else:
		dataSet = list(zip(totalStateBatch, totalActionBatch))
	saveFile = open(path, "wb")
	pickle.dump(dataSet, saveFile)


def loadData(path):
	pklFile = open(path, "rb")
	dataSet = pickle.load(pklFile)
	pklFile.close()
	return dataSet


def sampleData(data, batchSize, withReward=True):
	batch = random.sample(data, batchSize)
	if withReward:
		batchInput = [x for x, _, _ in batch]
		batchOutput1 = [y for _, y, _ in batch]
		batchOutput2 = [z for _, _, z in batch]
		reformatedBatch = (batchInput, batchOutput1, batchOutput2)
	else:
		batchInput = [x for x, _ in batch]
		batchOutput = [y for _, y in batch]
		reformatedBatch = (batchInput, batchOutput)
	return reformatedBatch


def prepareDataContinuousEnv():
	import continuousEnv as env
	xbound = [0, 180]
	ybound = [0, 180]
	vel = 1
	transitionFunction = env.TransitionFunction(xbound, ybound, vel)
	isTerminal = env.IsTerminal(vel+.5)
	reset = env.Reset(xbound, ybound)

	maxTimeStep = 4
	sampleTraj = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

	decay = 0.99
	rewardFunction = lambda state, action: -1
	accumulateRewards = AccumulateRewards(decay, rewardFunction)

	policy = env.OptimalPolicy(env.actionSpace)
	trajNum = 5
	path = "./continuous_data_with_reward.pkl"
	generateData(sampleTraj, accumulateRewards, policy, env.actionSpace, trajNum, path, withReward=True)

	data = loadData(path)
	print("{} data points in {}".format(len(data), path))


if __name__ == "__main__":
	prepareDataContinuousEnv()
