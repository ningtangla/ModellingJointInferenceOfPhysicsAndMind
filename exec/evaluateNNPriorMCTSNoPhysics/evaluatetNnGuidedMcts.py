
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

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
import time


import src.constrainedChasingEscapingEnv.reward as reward
from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from src.neuralNetwork.policyNet import GenerateModel, Train, restoreVariables

from src.constrainedChasingEscapingEnv.measurementFunctions import DistanceBetweenActualAndOptimalNextPosition
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory

import src.constrainedChasingEscapingEnv.envNoPhysics as env
from src.algorithms.mcts import MCTS, CalculateScore, selectGreedyAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from exec.evaluateNoPhysicsEnvWithRender import Render


def drawPerformanceLine(dataDf, axForDraw, trainSteps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(trainSteps))

#
class GetMCTS:
    def __init__(self, selectChild, rollout, backup, selectNextAction, getActionPriorFunction):
        self.selectChild = selectChild
        self.rollout = rollout
        self.backup = backup
        self.selectNextAction = selectNextAction
        self.getActionPriorFunction = getActionPriorFunction

    def __call__(self, numSimulations, trainedModel):
        actionPriorFunction = self.getActionPriorFunction(trainedModel)
        expand = Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit, actionPriorFunction))
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


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy, trialIndex):
        state = self.reset(trialIndex)
        while self.isTerminal(state):
            state = self.reset(trialIndex)
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


class GenerateTrajectories:
    def __init__(self, sampleTrajectory, getSheepPolicies, wolfPolicy, numTrials, getSavePath, trainedModels):
        self.sampleTrajectory = sampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.wolfPolicy = wolfPolicy
        self.numTrials = numTrials
        self.getSavePath = getSavePath
        self.trainedModels = trainedModels

    def __call__(self, oneConditionDf):
        startTime = time.time()
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]

        trainedModel = trainedModels[trainSteps]
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations, trainedModel)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)

        if not os.path.isfile(saveFileName):
            allTrajectories = [self.sampleTrajectory(policy, trial) for trial in range(self.numTrials)]
            pickleOut = open(saveFileName, 'wb')
            pickle.dump(allTrajectories, pickleOut)
            pickleOut.close()

        endTime = time.time()
        print("Time for policy {}, numSimulations {}, trainSteps {} = {}".format(sheepPolicyName, numSimulations,
                                                                                 trainSteps, (endTime - startTime)))

        return None


if __name__ == "__main__":
    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 30
    maxRunningSteps = 30
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = [0, 500, 1000, 10000]  # according to loss
    manipulatedVariables['sheepPolicyName'] = ['random', 'MCTS', 'NN', 'MCTSNN']
    manipulatedVariables['numSimulations'] = [50, 200]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    numOfAgent = 2
    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]

    minDistance = 25
    xBoundary = [0, 320]
    yBoundary = [0, 240]

    renderOn = False
    from pygame.color import THECOLORS
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPreyPos, getPredatorPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    actionMagnitude = 6
    wolfPolicy = HeatSeekingContinuesDeterministicPolicy(getPredatorPos, getPreyPos, actionMagnitude)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPriorUniformFunction = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    def sheepTransit(state, action): return transitionFunction(
        state, [action, wolfPolicy(state)])

    maxRolloutSteps = 5

    # reward function
    aliveBonus = 1 / maxRolloutSteps
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # random rollout policy
    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]  # [64]*3
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    model = generatePolicyNet(hiddenWidths)

    dataSetMaxRunningSteps = 30
    dataSetNumSimulations = 200
    dataSetNumTrials = 8000
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate}
    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)
    modelSavePaths = {trainSteps: getModelSavePath({'trainSteps': trainSteps}) for trainSteps in
                      manipulatedVariables['trainSteps']}
    trainedModels = {trainSteps: restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath) for
                     trainSteps, modelSavePath in modelSavePaths.items()}

    # wrapper function for expand
    uniformActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}
    getActionPriorUniformFunction = lambda trainedModel: uniformActionPrior
    getActionPriorNNFunction = lambda trainedModel: GetActionDistNeuralNet(actionSpace, trainedModel)

    # wrapper functions for sheep policies
    getMCTS = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorUniformFunction)
    getMCTSNN = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorNNFunction)
    getRandom = lambda numSimulations, trainedModel: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations, trainedModel: NeuralNetworkPolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNN': getMCTSNN}

    # sample trajectory
    initPositionList = [[env.samplePosition(xBoundary, yBoundary) for j in range(numOfAgent)]
                        for i in range(numTrials)]

    reset = env.FixedReset(initPositionList)
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, render, renderOn)

    # function to generate trajectories
    trajectoryDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, wolfPolicy, numTrials,
                                                getTrajectorySavePath, trainedModels)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, numTrials, measurementFunction=len)

    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    # plot the statistics
    fig = plt.figure()

    numColumns = len(manipulatedVariables['trainSteps'])
    plotCounter = 1

    for key, grp in statisticsDf.groupby('trainSteps'):
        grp.index = grp.index.droplevel('trainSteps')
        axForDraw = fig.add_subplot(1, numColumns, plotCounter)
        drawPerformanceLine(grp, axForDraw, key)
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()
