import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import time

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import CalculateScore, SelectChild, InitializeChildren, selectGreedyAction, RollOut, Expand, \
    MCTS, backup
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from src.neuralNetwork.policyNet import GenerateModel, restoreVariables
from src.constrainedChasingEscapingEnv.measurementFunctions import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory, \
    GetStateFromTrajectory


def drawPerformanceLine(dataDf, axForDraw, trainSteps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(trainSteps))

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
            allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]
            pickleOut = open(saveFileName, 'wb')
            pickle.dump(allTrajectories, pickleOut)
            pickleOut.close()

        endTime = time.time()
        print("Time for policy {}, numSimulations {}, trainSteps {} = {}".format(sheepPolicyName, numSimulations,
                                                                                 trainSteps, (endTime-startTime)))

        return None


if __name__ == "__main__":
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 100
    maxRunningSteps = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = [1000, 5000, 10000, 15000]
    manipulatedVariables['sheepPolicyName'] = ['random', 'MCTS', 'NN', 'MCTSNN']
    manipulatedVariables['numSimulations'] = [10, 50, 100, 150, 200]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    qPosInit = (0, 0, 4, 0)

    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 6
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 12
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    dataSetMaxRunningSteps = 10
    dataSetNumSimulations = 100
    dataSetNumTrials = 1500
    dataSetQPosInit = (0, 0, 8, 0)
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate}
    modelSaveDirectory = "../../data/testMCTSUniformVsNNPriorSheepChaseWolfMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)
    modelSavePaths = {trainSteps: getModelSavePath({'trainSteps': trainSteps}) for trainSteps in
                      manipulatedVariables['trainSteps']}
    trainedModels = {trainSteps: restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath) for
                     trainSteps, modelSavePath in modelSavePaths.items()}

    # wrapper functions for action prior
    uniformActionPrior = lambda state: {action: 1/len(actionSpace) for action in actionSpace}
    getActionPriorUniformFunction = lambda trainedModel: uniformActionPrior
    getActionPriorNNFunction = lambda trainedModel: GetActionDistNeuralNet(actionSpace, trainedModel)

    # wrapper functions for sheep policies
    getMCTS = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorUniformFunction)
    getMCTSNN = GetMCTS(selectChild, rollout, backup, selectGreedyAction, getActionPriorNNFunction)
    getRandom = lambda numSimulations, trainedModel: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations, trainedModel: NeuralNetworkPolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNN': getMCTSNN}

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # function to generate trajectories
    trajectoryDirectory = "../../data/testMCTSUniformVsNNPriorSheepChaseWolfMujoco/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
                                                getTrajectorySavePath, trainedModels)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # measurement Function
    initTimeStep = 0
    stateIndex = 0
    getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    getOptimalAction = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getWolfXPos, getSheepXPos)
    computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction, sheepTransit, getSheepXPos)
    measurementTimeStep = 1
    getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepXPos, getNextStateFromTrajectory)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos, getPosAtNextStepFromTrajectory)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, numTrials, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print(statisticsDf)

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