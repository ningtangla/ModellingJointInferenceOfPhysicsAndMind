import numpy as np
import pygame as pg
import os
import time
from AnalyticGeometryFunctions import computeAngleBetweenVectors
# init variables
gridSize = 10
actionSpace = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
stateDim = 4
actionDim = 8


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


def checkBound(state, gridSize):
	xPos, yPos = state
	if xPos >= gridSize or xPos < 0:
		return False
	elif yPos >= gridSize or yPos < 0:
		return False
	return True


class TransitionFunction():
	def __init__(self, gridSize):
		self.gridSize = gridSize

	def __call__(self, state, action):
		newState = np.array(state) + np.array(action)
		newState = tuple(newState)
		if checkBound(newState, self.gridSize):
			return newState
		return state


class IsTerminal():
	def __init__(self):
		return

	def __call__(self, state, targetPosition):
		if (np.array(state) == targetPosition).all():
			return True
		return False


class Reset():
	def __init__(self, actionSpace, agentStateSpace, targetStateSpace):
		self.actionSpace = actionSpace
		self.agentStateSpace = agentStateSpace
		self.targetStateSpace = targetStateSpace

	def __call__(self):
		initialAgentState = self.agentStateSpace[np.random.randint(0, len(self.agentStateSpace))]
		targetPosition = self.targetStateSpace[np.random.randint(0, len(self.targetStateSpace))]
		return initialAgentState, targetPosition


# class Render:
# 	def __init__(self, gridSize, action):
# 		pg.init()
# 		self.windowsize = [255 * 3, 255 * 3]
# 		self.screen = pg.display.set_mode(self.windowsize)
# 		pg.display.set_caption("Grid")
# 		self.clock = pg.time.Clock()
# 		self.finish = False
# 		self.margin = 5
# 		self.width = gridSize
# 		self.height = gridSize
# 		self.action = action
# 		self.drawingConstant = 5
#
# 	def __call__(self):
# 		done = False
# 		while not done:
# 			for event in pg.event.get():
# 				if event.type == 4:
# 					done = True
# 			self.screen.fill(BLACK)
# 			for action in self.action:
# 				original = self.player
# 				pg.draw.rect(self.screen, color['space'], [
# 				self.drawingConstant *
# 				((self.margin + self.width) *
# 				 (original[0] + 1) + self.margin), self.drawingConstant *
# 				((self.margin + self.height) *
# 				 (self.height-original[1] + 1) + self.margin),
# 				self.drawingConstant * self.width,
# 				self.drawingConstant * self.height
# 				])
# 				for y in range(0, self.height):
# 					for x in range(0, self.width):
# 						object = self.grid[y][x]
# 						pg.draw.rect(self.screen, color[object], [
# 						self.drawingConstant *
# 							((self.margin + self.width) *
# 				 (x + 1) + self.margin), self.drawingConstant *
# 				((self.margin + self.height) *
# 				 (self.height-y + 1) + self.margin),
# 				self.drawingConstant * self.width,
# 				self.drawingConstant * self.height
# 				])
# 				if action == original:
# 					continue
# 				self.player = action
# 				pg.draw.rect(self.screen, color['player'], [
# 					self.drawingConstant *
# 				((self.margin + self.width) *
# 				 (self.player[0] + 1) + self.margin), self.drawingConstant *
# 				((self.margin + self.height) *
# 				 (self.height - self.player[1] + 1) + self.margin),
# 				self.drawingConstant * self.width,
# 				self.drawingConstant * self.height
# 				])
#
# 		self.clock.tick(60)
# 		pg.display.flip()
# 		time.sleep(0.5)
# 		pg.quit()