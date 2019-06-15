import numpy as np
import pygame as pg
import mcts
# import stochasticPolicyValueNet as net
import policyValueNet as net
import dataTools
import sheepEscapingEnv as env
import sheepEscapingEnvRender as envRender
import evaluateSheepEscapingPolicy as eval
import os
import subprocess


def makeVideo(videoName, path):
	absolutePath = os.path.join(os.getcwd(), path)
	os.chdir(absolutePath)
	fps = 5
	crf = 25
	resolution = '1920x1080'
	cmd = 'ffmpeg -r {} -s {} -i %d.png -vcodec libx264 -crf {} -pix_fmt yuv420p {}'.format(fps, resolution, crf, videoName).split(" ")
	subprocess.call(cmd)
	if os.path.exists(videoName):
		[os.remove(file) if file.endswith(".png") else 0 for file in os.listdir(os.getcwd())]
	else:
		print("Demo generate Failed, needs to be done manually")


class SheepEscapingDemo:
	def __init__(self, trajNum, maxTrajLen, renderOn, savePath=None, saveVideo=False, seed=128):
		self.trajNum = trajNum
		self.maxTrajLen = maxTrajLen
		self.renderOn = renderOn
		self.savePath = savePath
		self.saveVideo = saveVideo
		self.seed = seed

	def __call__(self, policy, useActionDist):
		xBoundary = env.xBoundary
		yBoundary = env.yBoundary
		actionSpace = env.actionSpace

		wolfHeatSeekingPolicy = env.WolfHeatSeekingPolicy(actionSpace)
		transition = env.TransitionFunction(xBoundary, yBoundary, env.vel, wolfHeatSeekingPolicy)
		isTerminal = env.IsTerminal(minDistance=env.vel + 5)
		reset = env.Reset(xBoundary, yBoundary, initialSeed=self.seed)

		sampleTraj = dataTools.SampleTrajectory(self.maxTrajLen, transition, isTerminal, reset, useActionDist)
		evaluate = eval.Evaluate(sampleTraj, trajNum)
		evalResults, demoStates = evaluate(policy)
		print(evalResults)

		if renderOn:
			extendedBound = 10
			screen = pg.display.set_mode([xBoundary[1], yBoundary[1] + extendedBound])
			render = envRender.Render(screen, self.savePath, 1)
			for trajIndex in range(len(demoStates)):
				for stepIndex in range(len(demoStates[trajIndex])):
					render(demoStates[trajIndex][stepIndex], trajIndex)
			if self.saveVideo:
				videoName = "mean_{}_{}_trajectories_nn_demo.mp4".format(evalResults['mean'], trajNum)
				makeVideo(videoName, self.savePath)

		return evalResults


if __name__ == "__main__":
	trajNum = 1
	maxTrajLen = 100
	renderOn = True
	demo = SheepEscapingDemo(trajNum, maxTrajLen, renderOn)

	nnDemo = False
	if nnDemo:
		modelPath = "savedModels/60000data_64x4_minibatch_100kIter_contState_actionDist"
		modelStructure = [64]*4
		generateModel = net.GenerateModelSeparateLastLayer(env.numStateSpace, env.numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
		model = generateModel(modelStructure)
		trainedModel = net.restoreVariables(model, modelPath)
		policy = lambda state: net.approximatePolicy(state, trainedModel, env.actionSpace)
		useActionDist = False
	else:
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
		useActionDist = True
		getPolicyOutput = mcts.getSoftmaxActionDist
		policy = mcts.MCTSPolicy(numSimulations, selectChild, expand, nodeValue, mcts.backup, getPolicyOutput)

	demo(policy, useActionDist)
