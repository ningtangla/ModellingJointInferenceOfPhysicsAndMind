import sys
import os
sys.path.append(os.path.join('..', '..', 'src', 'algorithms'))
sys.path.append(os.path.join('..', '..', 'src', 'sheepWolf'))
sys.path.append(os.path.join('..', '..', 'src'))
sys.path.append(os.path.join('..', '..', 'src', 'neuralNetwork'))
sys.path.append('..')

import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from envMujoco import Reset, IsTerminal, TransitionFunction
from mcts import CalculateScore, SelectChild, InitializeChildren, GetActionPrior, SelectNextAction, RollOut,\
HeuristicDistanceToTarget, Expand, MCTS, backup
from play import SampleTrajectory
import reward
from policiesFixed import stationaryAgentPolicy
from evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from policyNet import GenerateModel, Train, restoreVariables
from measurementFunctions import DistanceBetweenActualAndOptimalNextPosition
from sheepWolfWrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory, GetTrialTrajectoryFromDf


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
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)

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
    qPosInit = (-4, 0, 4, 0)

    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = 2
    numXPosEachAgent = 2
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex, numXPosEachAgent)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    selectNextAction = SelectNextAction(sheepTransit)

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    model = generatePolicyNet(hiddenWidths)

    dataSetMaxRunningSteps = 15
    dataSetNumSimulations = 200
    dataSetNumTrials = 100
    dataSetQPosInit = (-4, 0, 4, 0)
    trainSteps = 50000
    modelTrainConditions = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                            'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                            'learnRate': learningRate, 'trainSteps': trainSteps}
    modelSaveDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)
    modelSavePath = getModelSavePath(modelTrainConditions)

    trainedModel = restoreVariables(model, modelSavePath)
    print("restored saved model")

    # wrapper function for action prior
    initState = reset()
    getActionPriorUniform = GetActionPrior(actionSpace)
    getActionPriorNNAllSteps = GetActionDistNeuralNet(actionSpace, trainedModel)
    getActionPriorNNFirstStep = GetNonUniformPriorAtSpecificState(getActionPriorNNAllSteps,
                                                                         getActionPriorUniform, initState)

    # wrapper functions for sheep policies
    getMCTS = GetMCTS(selectChild, rollout, backup, selectNextAction, getActionPriorUniform)
    getMCTSNNFirstStep = GetMCTS(selectChild, rollout, backup, selectNextAction, getActionPriorNNFirstStep)
    getMCTSNNAllSteps = GetMCTS(selectChild, rollout, backup, selectNextAction, getActionPriorNNAllSteps)
    getRandom = lambda numSimulations: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations: NeuralNetworkPolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNNFirstStep': getMCTSNNFirstStep,
                        'MCTSNNAllSteps': getMCTSNNAllSteps}

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # function to generate trajectories
    trajectoryDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
                                                getTrajectorySavePath)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # measurement Function
    optimalAction = (10, 0)
    optimalNextState = sheepTransit(initState, optimalAction)
    optimalNextPosition = getSheepXPos(optimalNextState)
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
