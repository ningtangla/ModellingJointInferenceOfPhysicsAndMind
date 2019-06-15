import numpy as np
from AnalyticGeometryFunctions import computeVectorNorm, computeAngleBetweenVectors
# import pygame as pg
# from anytree import AnyNode as Node
# import os
import dataTools

numStateSpace = 4
actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
numActionSpace = len(actionSpace)
xBoundary = [0, 180]
yBoundary = [0, 180]
vel = 20


class CheckBoundaryAndAdjust():
	def __init__(self, xBoundary, yBoundary):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.xMin, self.xMax = xBoundary
		self.yMin, self.yMax = yBoundary

	def __call__(self, position):
		if position[0] >= self.xMax:
			position[0] = 2 * self.xMax - position[0]
		if position[0] <= self.xMin:
			position[0] = 2 * self.xMin - position[0]
		if position[1] >= self.yMax:
			position[1] = 2 * self.yMax - position[1]
		if position[1] <= self.yMin:
			position[1] = 2 * self.yMin - position[1]
		return position


def getEachState(state):
	return state[:2], state[2:]


class TransitionFunction():
	def __init__(self, xBoundary, yBoundary, velocity, wolfPolicy):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.velocity = velocity
		self.wolfPolicy = wolfPolicy
		self.checkBound = CheckBoundaryAndAdjust(xBoundary, yBoundary)

	def __call__(self, state, action):
		oldSheepPos, oldWolfPos = getEachState(state)
		# wolf
		wolfAction = self.wolfPolicy(state)
		wolfActionMagnitude = computeVectorNorm(np.array(wolfAction))
		modifiedWolfAction = np.array(wolfAction) * 0.95*self.velocity / wolfActionMagnitude
		newWolfPos = np.array(oldWolfPos) + modifiedWolfAction
		# sheep
		sheepActionMagnitude = computeVectorNorm(np.array(action))
		modifiedSheepAction = np.array(action) * self.velocity / sheepActionMagnitude
		newSheepPos = np.array(oldSheepPos) + modifiedSheepAction
		sheepPos = self.checkBound(newSheepPos)
		wolfPos = self.checkBound(newWolfPos)
		return np.concatenate([sheepPos, wolfPos])


class IsTerminal():
	def __init__(self, minDistance):
		self.minDistance = minDistance

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(sheepState) - np.array(wolfState)
		relativeDistance = computeVectorNorm(relativeVector)
		if relativeDistance <= self.minDistance:
			return True
		return False


class Reset():
	def __init__(self, xBoundary, yBoundary, initialSeed=None, maxInitDist=np.inf):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.checkBound = CheckBoundaryAndAdjust(xBoundary, yBoundary)
		self.seed = initialSeed
		self.maxInitDist = maxInitDist

	def __call__(self):
		if self.seed is not None:
			np.random.seed(self.seed)
			self.seed = self.seed + 1
		xMin, xMax = self.xBoundary
		yMin, yMax = self.yBoundary
		initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
		initialDistance = computeVectorNorm(targetPosition - initialAgentState)
		while not ((self.checkBound(initialAgentState) == initialAgentState).all() and (self.checkBound(targetPosition) == targetPosition).all() and initialDistance <= self.maxInitDist):
			initialAgentState = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
			targetPosition = np.array([np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax)])
			initialDistance = computeVectorNorm(targetPosition - initialAgentState)
		return np.concatenate([initialAgentState, targetPosition])


class ResetWithinDataSet():
	def __init__(self, xBoundary, yBoundary, dataSet, minInitDist=0):
		self.xBoundary = xBoundary
		self.yBoundary = yBoundary
		self.checkBound = CheckBoundaryAndAdjust(xBoundary, yBoundary)
		self.dataSet = dataSet
		self.minInitDist = minInitDist

	def __call__(self):
		point = dataTools.sampleData(self.dataSet, 1)
		state, action, value = point
		sheep, wolf = getEachState(state[0])
		distance = computeVectorNorm(np.array(sheep) - np.array(wolf))
		while not (distance >= self.minInitDist):
			point = dataTools.sampleData(self.dataSet, 1)
			state, action, value = point
			sheep, wolf = getEachState(state[0])
			distance = computeVectorNorm(np.array(sheep) - np.array(wolf))
		return state[0]


class WolfHeatSeekingPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(sheepState) - np.array(wolfState)
		angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
							   np.array(self.actionSpace)}
		action = angleBetweenVectors[min(angleBetweenVectors.keys())]
		return action


class SheepNaiveEscapingPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		sheepState, wolfState = getEachState(state)
		relativeVector = np.array(wolfState) - np.array(sheepState)
		angleBetweenVectors = {computeAngleBetweenVectors(relativeVector, action): action for action in
							   np.array(self.actionSpace)}
		action = angleBetweenVectors[max(angleBetweenVectors.keys())]
		return action


class SheepRandomPolicy:
	def __init__(self, actionSpace):
		self.actionSpace = actionSpace

	def __call__(self, state):
		actionIndex = np.random.randint(len(actionSpace))
		action = actionSpace[actionIndex]
		return action