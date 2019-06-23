
import sys
sys.path.append('../')
sys.path.append('../../src')
sys.path.append('../../src/algorithms')
sys.path.append('../../src/constrainedChasingEscapingEnv')
sys.path.append('../../src/neuralNetwork')

import os
import numpy as np
import pickle
import random
import pygame as pg
import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
import pickle
import pandas as pd
from matplotlib import pyplot as plt


import reward
from evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from policyNet import GenerateModel, Train, restoreVariables
from measurementFunctions import DistanceBetweenActualAndOptimalNextPosition, computeDistance
from wrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory

import envNoPhysics as env
from algorithms.mcts import MCTS, CalculateScore, selectGreedyAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren

import reward
from evaluationFunctions import GetSavePath
from policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from analyticGeometryFunctions import computeAngleBetweenVectors


class Render():
    def __init__(self, numOfAgent, numPosEachAgent, positionIndex, screen, screenColor, circleColorList, circleSize):
        self.numOfAgent = numOfAgent
        self.numPosEachAgent = numPosEachAgent
        self.positionIndex = positionIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)

            for i in range(self.numOfAgent):
                agentPos = state[i][self.positionIndex:self.positionIndex +
                                    self.numPosEachAgent]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(1)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None))
                break

            if self.renderOn:
                self.render(state)
            action = policy(state)
            trajectory.append((state, action))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class GetTrialTrajectoryFromDf:
    def __init__(self, trialIndex):
        self.trialIndex = trialIndex

    def __call__(self, dataFrame):
        trajectory = dataFrame.values[self.trialIndex]
        return trajectory


def drawPerformanceLine(dataDf, axForDraw):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='Distance Between Optimal And Actual First Step'
                                                                      ' of Sheep')


class GetMCTS:
    def __init__(self, selectChild, rollout, backup, selectNextAction, actionPriorFunction):
        self.selectChild = selectChild
        self.rollout = rollout
        self.backup = backup
        self.selectNextAction = selectNextAction
        self.actionPriorFunction = actionPriorFunction

    def __call__(self, numSimulations):
        expand = Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit, self.actionPriorFunction))
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectGreedyAction)

        return mcts


class GetActionDistNeuralNet:
    def __init__(self, actionSpace, model):
        self.actionSpace = actionSpace
        self.model = model

    def __call__(self, state):
        stateFlat = np.asarray(state).flatten()
        graph = self.model.graph
        actionDistribution_ = graph.get_collection_ref("actionDistribution")[0]
        state_ = graph.get_collection_ref("inputs")[0]
        actionDistribution = self.model.run(actionDistribution_, feed_dict={state_: [stateFlat]})[0]
        actionDistributionDict = dict(zip(self.actionSpace, actionDistribution))

        return actionDistributionDict


class GetNonUniformPriorAtSpecificState:
    def __init__(self, getNonUniformPrior, getUniformPrior, specificState):
        self.getNonUniformPrior = getNonUniformPrior
        self.getUniformPrior = getUniformPrior
        self.specificState = specificState

    def __call__(self, currentState):
        if (currentState == self.specificState).all():
            actionPrior = self.getNonUniformPrior(currentState)
        else:
            actionPrior = self.getUniformPrior(currentState)

        return actionPrior


class NeuralNetworkPolicy:
    def __init__(self, model, actionSpace):
        self.model = model
        self.actionSpace = actionSpace

    def __call__(self, state):
        stateFlat = np.asarray(state).flatten()
        graph = self.model.graph
        actionDistribution_ = graph.get_collection_ref("actionDistribution")[0]
        state_ = graph.get_collection_ref("inputs")[0]
        actionDistribution = self.model.run(actionDistribution_, feed_dict={state_: [stateFlat]})[0]
        maxIndex = np.argwhere(actionDistribution == np.max(actionDistribution)).flatten()
        actionIndex = np.random.choice(maxIndex)
        action = self.actionSpace[actionIndex]

        return action


class GenerateTrajectories:
    def __init__(self, sampleTrajectory, getSheepPolicies, wolfPolicy, numTrials, getSavePath):
        self.sampleTrajectory = sampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.wolfPolicy = wolfPolicy
        self.numTrials = numTrials
        self.getSavePath = getSavePath

    def __call__(self, oneConditionDf):
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]

        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)
        pickle_in = open(saveFileName, 'wb')
        pickle.dump(allTrajectories, pickle_in)
        pickle_in.close()

        return allTrajectories


if __name__ == "__main__":
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 100
    maxRunningSteps = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepPolicyName'] = ['random', 'MCTS', 'NN', 'MCTSNNFirstStep', 'MCTSNNAllSteps']
    manipulatedVariables['numSimulations'] = [5, 25, 50, 100, 250]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    numOfAgent = 2
    numOfOneAgentState = 2
    maxRunningSteps = 200

    sheepId = 0
    wolfId = 1
    positionIndex = 0
    numPosEachAgent = 2
    minDistance = 25

    xBoundary = [0, 640]
    yBoundary = [0, 480]

    initPosition = np.array([[30, 30], [200, 200]])
    # initPosition = np.array([[np.random.uniform(xBoundary[0], xBoundary[1]), np.random.uniform(yBoundary[0], yBoundary[1])], [np.random.uniform(xBoundary[0], xBoundary[1]), np.random.uniform(yBoundary[0], yBoundary[1])]])
    initPositionNoise = [0, 0]

    renderOn = True
    from pygame.color import THECOLORS
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, numOfOneAgentState, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex, numPosEachAgent)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex, numPosEachAgent)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPreyPos, getPredatorPos, minDistance, computeDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(numOfAgent, initPosition, initPositionNoise)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getPreyPos, getPredatorPos, computeAngleBetweenVectors)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    def sheepTransit(state, action): return transitionFunction(
        state, [action, wolfPolicy(state)])

    # reward function
    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # initialize children; expand
    initializeChildren = InitializeChildren(
        actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    model = generatePolicyNet(hiddenWidths)

    dataSetMaxRunningSteps = 200
    dataSetNumSimulations = 10
    dataSetNumTrials = 10
    dataSetInitPosition = np.array([[30, 30], [200, 200]])
    trainSteps = 10000
    modelTrainConditions = {'maxRunningSteps': dataSetMaxRunningSteps, 'posInit': dataSetInitPosition,
                            'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                            'learnRate': learningRate, 'trainSteps': trainSteps}
    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)
    modelSavePath = getModelSavePath(modelTrainConditions)

    trainedModel = restoreVariables(model, modelSavePath)
    print("restored saved model")

    # wrapper function for action prior
    initState = reset()
    getActionPriorUniform = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}
    getActionPriorNNAllSteps = GetActionDistNeuralNet(actionSpace, trainedModel)
    getActionPriorNNFirstStep = GetNonUniformPriorAtSpecificState(getActionPriorNNAllSteps,
                                                                  getActionPriorUniform, initState)

    # wrapper functions for sheep policies
    getMCTS = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorUniform)
    getMCTSNNFirstStep = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorNNFirstStep)
    getMCTSNNAllSteps = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorNNAllSteps)
    getRandom = lambda numSimulations: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations: NeuralNetworkPolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNNFirstStep': getMCTSNNFirstStep,
                        'MCTSNNAllSteps': getMCTSNNAllSteps}

    # sample trajectory
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, render, renderOn)

    # function to generate trajectories
    trajectoryDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'posInit': initPosition}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, wolfPolicy, numTrials,
                                                getTrajectorySavePath)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # measurement Function
    optimalAction = (10, 0)
    optimalNextState = sheepTransit(initState, optimalAction)
    optimalNextPosition = getPreyPos(optimalNextState)
    measurementTimeStep = 1
    stateIndex = 0
    getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(measurementTimeStep, stateIndex, getSheepXPos)
    getFirstTrajectoryFromDf = GetTrialTrajectoryFromDf(0)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(optimalNextPosition, getPosAtNextStepFromTrajectory)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, numTrials, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()

    axForDraw = fig.add_subplot(1, 1, 1)
    drawPerformanceLine(statisticsDf, axForDraw)

    plt.legend(loc='best')

    plt.show()
