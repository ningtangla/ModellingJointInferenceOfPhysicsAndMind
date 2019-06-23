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
import time
from multiprocessing import Pool
import multiprocessing as mp

from envMujoco import Reset, IsTerminal, TransitionFunction
from mcts import CalculateScore, SelectChild, InitializeChildren, GetActionPrior, selectNextAction, RollOut,\
HeuristicDistanceToTarget, Expand, MCTS, backup
from play import SampleTrajectory
import reward
from policiesFixed import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from policyNet import GenerateModel, Train, restoreVariables
from measurementFunctions import DistanceBetweenActualAndOptimalNextPosition, ComputeOptimalNextPos
from sheepWolfWrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory, GetStateFromTrajectory


def drawPerformanceLine(dataDf, axForDraw, steps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        # grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(steps))
        grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(steps))
        axForDraw.set_ylim([0, 0.4])


# class RunSampleTrajectory:
#     def __init__(self, getSampleTrajectory, policy):
#         self.getSampleTrajectory = getSampleTrajectory
#         self.policy = policy
#
#     def __call__(self, qPosInit):
#         sampleTrajectory = self.getSampleTrajectory(qPosInit)
#         trajectory = sampleTrajectory(self.policy)
#
#         return trajectory

def runSampleTrajectory(qPosInit, getSampleTrajectory, policy):
    sampleTrajectory = getSampleTrajectory(qPosInit)
    trajectory = sampleTrajectory(policy)

    return trajectory


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
    def __init__(self, getSampleTrajectory, getSheepPolicies, wolfPolicy, numTrials, getSavePath, trainedModels,
                 getReset, allQPosInit, pool):
        self.getSampleTrajectory = getSampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.wolfPolicy = wolfPolicy
        self.numTrials = numTrials
        self.getSavePath = getSavePath
        self.trainedModels = trainedModels
        self.getReset = getReset
        self.allQPosInit = allQPosInit
        self.pool = pool

    def __call__(self, oneConditionDf):
        startTime = time.time()
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]

        trainedModel = self.trainedModels[trainSteps]
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations, trainedModel)
        # policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]
        def policy(state):
            return [sheepPolicy(state), self.wolfPolicy(state)]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)

        if not os.path.isfile(saveFileName):
            # allTrajectories = self.pool.map(runSampleTrajectory, self.allQPosInit)
            allTrajectories = [self.pool.apply(runSampleTrajectory, args=(qPosInit, self.getSampleTrajectory, policy))
                               for qPosInit in self.allQPosInit]
            print("ALL TRAJECTORIES: ", allTrajectories)
            # pickleOut = open(saveFileName, 'wb')
            # pickle.dump(allTrajectories, pickleOut)
            # pickleOut.close()

        endTime = time.time()
        print("Time for policy {}, numSimulations {}, trainSteps {} = {}".format(sheepPolicyName, numSimulations,
                                                                                 trainSteps, (endTime-startTime)))

        return None


def main():
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 2#50
    maxRunningSteps = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = [0, 10, 20, 50]
    manipulatedVariables['sheepPolicyName'] = ['NN', 'MCTSNN', 'MCTS']
    manipulatedVariables['numSimulations'] = [1, 2]#[50, 200, 800]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # generate a set of initial positions to maintain consistency across all conditions
    allQPosInit = [np.random.uniform(0, 9.7, 4) for trial in range(numTrials)]

    # functions for MCTS
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    getReset = lambda qPosInit: Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)        ###########

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
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

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
    dataSetNumSimulations = 75
    dataSetNumTrials = 1500
    dataSetQPosInit = (0, 0, 0, 0)
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate}
    modelSaveDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)
    modelSavePaths = {trainSteps: getModelSavePath({'trainSteps': trainSteps}) for trainSteps in
                      manipulatedVariables['trainSteps']}
    trainedModels = {trainSteps: restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath) for
                     trainSteps, modelSavePath in modelSavePaths.items()}

    # wrapper function for expand
    getExpandUniformPrior = lambda trainedModel: Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit,
                                                                                       GetActionPrior(actionSpace)))
    getExpandNNPrior = lambda trainedModel: Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit,
                                                                                  GetActionDistNeuralNet(actionSpace,
                                                                                                         trainedModel)))

    # wrapper functions for sheep policies
    getMCTS = lambda numSimulations, trainedModel: MCTS(numSimulations, selectChild,
                                                        getExpandUniformPrior(trainedModel), rollout, backup,
                                                        selectNextAction)
    getMCTSNN = lambda numSimulations, trainedModel: MCTS(numSimulations, selectChild,
                                                        getExpandNNPrior(trainedModel), rollout, backup,
                                                        selectNextAction)
    getRandom = lambda numSimulations, trainedModel: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations, trainedModel: NeuralNetworkPolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNN': getMCTSNN}

    # getSampleTrajectory = lambda qPosInit: SampleTrajectory(maxRunningSteps, transit, isTerminal, getReset(qPosInit))
    def getSampleTrajectory(qPosInit):
        return SampleTrajectory(maxRunningSteps, transit, isTerminal, getReset(qPosInit))

    # pool for multiprocessing
    pool = Pool(mp.cpu_count())

    # function to generate trajectories
    trajectoryDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'processing': 'multiCore'}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(getSampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
                                                getTrajectorySavePath, trainedModels, getReset, allQPosInit, pool)

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

    print('statisticsDf')
    print(statisticsDf)

    # plot the statistics
    fig = plt.figure()

    numColumns = len(manipulatedVariables['trainSteps'])
    numRows = 1
    plotCounter = 1

    for key, grp in statisticsDf.groupby('trainSteps'):
        grp.index = grp.index.droplevel('trainSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        drawPerformanceLine(grp, axForDraw, key)
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()