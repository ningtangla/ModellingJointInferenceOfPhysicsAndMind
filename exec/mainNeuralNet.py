# the problem with getSavePath is that it is somehow dependent on the order of the variables. So, for the same values
# of the parameters, you could get different fileNames

# I have used a slightly bad list comprehension in the wrapper function for single agent get trajectory

# in train neural network, I have assumed maxStepNum is the number of training steps

import sys
sys.path.append('../src/algorithms')
sys.path.append('../src/sheepWolf')
sys.path.append('../src')

import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
import pickle
import pandas as pd
import os

from envMujoco import Reset, IsTerminal, TransitionFunction
from mcts import CalculateScore, SelectChild, InitializeChildren, GetActionPrior, SelectNextAction, RollOut,\
HeuristicDistanceToTarget, Expand, MCTS, backup
from play import SampleTrajectory
import reward
from wrapperFunctions import GetAgentPos
from policiesFixed import stationaryAgentPolicy
from evaluationFunctions import GetSavePath
from prepareNeuralNetData import loadData
from neuralNetwork import GeneratePolicyNet
from supervisedLearning import Train

def flattenStates(trainingData):                                                                                        # can be made more generic by removing the specific numbers
    allStates = trainingData[0]
    allLabels = trainingData[1]
    allStatesFlattened = [state.flatten() for state in allStates]

    return [allStatesFlattened, allLabels]


class GetMcts:
    def __init__(self, selectChild, rollout, backup, selectNextAction):
        self.selectChild = selectChild
        self.rollout = rollout
        self.backup = backup
        self.selectNextAction = selectNextAction

    def __call__(self, numSimulations, actionPriorFunction):
        expand = Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit, actionPriorFunction))
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)

        return mcts


class GetActionDistNeuralNet:
    def __init__(self, actionSpace, model):
        self.actionSpace = actionSpace
        self.model = model

    def __call__(self, state):
        graph = self.model.graph
        actionDistributionNode = graph.get_collection_ref("actionDistribution")
        actionDistributionOneHot = actionDistributionNode.eval(feed_dict=state)
        actionDistribution = dict(zip(self.actionSpace, actionDistributionOneHot))

        return actionDistribution


class GetNonUniformPriorAtSpecificState:
    def __init__(self, getNonUniformPrior, getUniformPrior, specificState):
        self.getNonUniformPrior = getNonUniformPrior
        self.getUniformPrior = getUniformPrior
        self.specificState = specificState

    def __call__(self, currentState):
        if currentState == self.specificState:
            actionPrior = self.getNonUniformPrior(currentState)
        else:
            actionPrior = self.getUniformPrior(currentState)

        return actionPrior


class NeuralNetworkPolicy:                                                                                              # the wrapper function allows me to have prior as an input. So I can think of using the prior instead of evaluating the model. But finding max. values with dictionaries is tricky
    def __init__(self, model):
        self.model = model

    def __call__(self, state):
        graph = self.model.graph
        actionDistributionNode = graph.get_collection_ref("actionDistribution")
        actionDistributionOneHot = actionDistributionNode.eval(feed_dict=state)
        maxIndex = np.argwhere(actionDistributionOneHot == np.max(actionDistributionOneHot)).flatten()
        action = np.random.choice(maxIndex)

        return action


class GenerateTrajectories:
    def __init__(self, maxRunningStepsForPolicy, getSampleTrajectory, getSheepPolicies, actionPriorFunctionForPolicy,
                 wolfPolicy, numTrials, getSavePath):
        self.maxRunningStepsForPolicy = maxRunningStepsForPolicy
        self.getSampleTrajectory = getSampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.actionPriorFunctionForPolicy = actionPriorFunctionForPolicy
        self.wolfPolicy = wolfPolicy
        self.numTrials = numTrials
        self.getSavePath = getSavePath

    def __call__(self, oneConditionDf):
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]

        maxRunningSteps = self.maxRunningStepsForPolicy[sheepPolicyName]
        sampleTrajectory = self.getSampleTrajectory(maxRunningSteps)

        actionPriorFunction = self.actionPriorFunctionForPolicy[sheepPolicyName]
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations, actionPriorFunction)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]

        saveFileName = self.getSavePath(oneConditionDf)
        pickle_in = open(saveFileName, 'wb')
        pickle.dump(allTrajectories, pickle_in)
        pickle_in.close()

        return allTrajectories


if __name__ == "__main__":
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables
    numTrials = 5
    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepPolicyName'] = ['random', 'mcts', 'nn', 'mctsNnFirstStep', 'mctsNnEveryStep']
    manipulatedVariables['numSimulations'] = [1, 2, 3, 4]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    qPosInit = [-4, 0, 4, 0]

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
    getSheepXPos = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, killzoneRadius, getSheepXPos, getWolfXPos)

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

    # Train neural network model
    numStateSpace = 12
    numActionSpace = len(actionSpace)

    dataSetDirectory = "../data/testNNPriorMCTSMujoco/trajectories"
    dataSetExtension = 'pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 15
    dataSetNumSimulations = 200
    dataSetNumTrials = 1
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': qPosInit, 'numSimulations': dataSetNumSimulations,
                          'numTrials': dataSetNumTrials}
    dataSetPath = getDataSetPath(dataSetConditionVariables)

    dataSetTrajectories = loadData(dataSetPath)
    random.shuffle(dataSetTrajectories)
    trainingData = ([state for state, action in dataSetTrajectories], [action for state, action in dataSetTrajectories])
    trainingDataFlat = flattenStates(trainingData)

    learningRate = 0.0001
    regularizationFactor = 1e-4
    generatePolicyNet = GeneratePolicyNet(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    modelToTrain = generatePolicyNet(2, 128)

    maxStepNum = 50000
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    train = Train(maxStepNum, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                     summaryOn=False, testData=None)

    trainedModel = train(modelToTrain, trainingDataFlat)

    # Neural Network Policy
    neuralNetworkPolicy = NeuralNetworkPolicy(trainedModel)

    # wrapper functions for sheep policies
    getMcts = GetMcts(selectChild, rollout, backup, selectNextAction)
    getRandom = lambda numSimulations, actionPriorFunction: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNn = lambda numSimulations, actionPriorFunction: neuralNetworkPolicy
    getSheepPolicies = {'mcts': getMcts, 'random': getRandom, 'nn': getNn, 'mctsNnFirstStep': getMcts,
                        'mctsNnEveryStep': getMcts}

    # wrapper function for action prior
    initState = reset()
    getActionPriorFunctionUniform = lambda model: GetActionPrior(actionSpace)
    getActionPriorFunctionNeuralNetAllSteps = lambda model: GetActionDistNeuralNet(actionSpace, model)
    getActionPriorFunctionNeuralNetFirstStep = lambda model: GetNonUniformPriorAtSpecificState(
        getActionPriorFunctionNeuralNetAllSteps(model), getActionPriorFunctionUniform(model), initState)

    # sample trajectory
    maxRunningStepsForPolicy = {'mcts': 15, 'random': 1, 'nn': 1, 'mctsNnFirstStep': 1, 'mctsNnEveryStep': 1}
    getSampleTrajectory = lambda maxRunningSteps: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # function to generate trajectories
    actionPriorFunctionForPolicy = {'mcts': getActionPriorFunctionUniform, 'random': getActionPriorFunctionUniform,
                                    'nn': getActionPriorFunctionUniform,
                                    'mctsNnFirstStep': getActionPriorFunctionNeuralNetFirstStep,
                                    'mctsNnEveryStep': getActionPriorFunctionNeuralNetAllSteps}


    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)

    extension = 'pickle'
    getSavePath = GetSavePath(dataDirectory, extension)

    generateTrajectories = GenerateTrajectories(maxRunningStepsForPolicy, getSampleTrajectory, getSheepPolicies,
                                                actionPriorFunctionForPolicy, stationaryAgentPolicy, numTrials, getSavePath)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getSavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, numTrials, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
