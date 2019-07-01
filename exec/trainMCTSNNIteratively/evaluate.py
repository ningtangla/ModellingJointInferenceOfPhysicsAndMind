import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import time

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import CalculateScore, SelectChild, InitializeChildren, selectGreedyAction, Expand, MCTS, \
    backup, RollOut
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from src.neuralNetwork.policyValueNet import GenerateModelSeparateLastLayer, restoreVariables, ApproximateActionPrior, \
    ApproximateValueFunction, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measurementFunctions import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory, \
    GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.wrappers import getStateFromNode, GetApproximateValueFromNode
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete


def drawPerformanceLine(dataDf, axForDraw, numSimulations, trainSteps):
    for key, grp in dataDf.groupby('modelLearnRate'):
        grp.index = grp.index.droplevel('modelLearnRate')
        grp.plot(ax=axForDraw, title='numSimulations={}, trainSteps={}'.format(numSimulations, trainSteps), y='mean',
                 yerr='std', marker='o', label='learnRate={}'.format(key))


class RestoreNNModel:
    def __init__(self, getModelSavePath, NNModel, restoreVariables):
        self.getModelSavePath = getModelSavePath
        self.NNmodel = NNModel
        self.restoreVariables = restoreVariables

    def __call__(self, iteration, learnRate):
        modelPath = self.getModelSavePath({'iteration': iteration, 'learnRate': learnRate})
        restoredNNModel = self.restoreVariables(self.NNmodel, modelPath)

        return restoredNNModel


def saveData(data, path):
    pickleOut = open(path, 'wb')
    pickle.dump(data, pickleOut)
    pickleOut.close()


class GetPolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, NNModel, numSimulations):
        sheepPolicy = self.getSheepPolicy(NNModel, numSimulations)
        wolfPolicy = self.getWolfPolicy(NNModel, numSimulations)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, getSampleTrajectory, numTrials, getSavePath, restoreNNModel, getPolicy, saveData):
        self.getSampleTrajectory = getSampleTrajectory
        self.numTrials = numTrials
        self.getSavePath = getSavePath
        self.restoreNNModel = restoreNNModel
        self.getPolicy = getPolicy
        self.saveData = saveData

    def __call__(self, oneConditionDf):
        startTime = time.time()
        iteration = oneConditionDf.index.get_level_values('iteration')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        learnRate = oneConditionDf.index.get_level_values('modelLearnRate')[0]

        NNModel = self.restoreNNModel(iteration, learnRate)
        policy = self.getPolicy(NNModel, numSimulations)

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)

        if not os.path.isfile(saveFileName):
            allSampleTrajectories = [self.getSampleTrajectory(trial) for trial in range(self.numTrials)]
            trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
            self.saveData(trajectories, saveFileName)

        endTime = time.time()
        print("Time for iteration {}, numSimulations {} = {}".format(iteration, numSimulations, (endTime-startTime)))

        return None


def main():
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 50
    maxRunningSteps = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['iteration'] = [0, 50, 200, 400, 800, 999]
    manipulatedVariables['numSimulations'] = [75]
    manipulatedVariables['modelLearnRate'] = [1e-10, 1e-6, 1e-8]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # generate a set of starting conditions to maintain consistency across all the conditions
    allQPosInit = np.random.uniform(-9.7, 9.7, (numTrials, 4))
    # pickleIn = open('allQPosInit.pickle', 'rb')
    # allQPosInit = pickle.load(pickleIn)
    # pickleIn.close()

    # functions for MCTS
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0

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

    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    aliveBonus = -0.05
    deathPenalty = 1
    reward = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getSheepXPos, getWolfXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, reward, isTerminal, rolloutHeuristic)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network model
    numStateSpace = 12
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    # generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, 0.0001, regularizationFactor)     # temporary modification

    trainMaxRunningSteps = 10
    trainQPosInit = (0, 0, 0, 0)
    trainQPosInitNoise = 9.7
    trainNumSimulations = 75
    trainSteps = 50
    trainNumTrialsEachIteration = 1
    trainFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'trainSteps': trainSteps, 'qPosInit': trainQPosInit,
                            'qPosInitNoise': trainQPosInitNoise, 'numSimulations': trainNumSimulations,
                            'numTrialsEachIteration': trainNumTrialsEachIteration}  # 'learnRate': learningRate
    NNModelSaveDirectory = "../../data/trainMCTSNNIteratively/trainedNNModels"
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)

    # wrapper function for expand
    approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, sheepTransit, approximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))

    # wrapper function for policy
    getApproximateValue = lambda NNModel: GetApproximateValueFromNode(getStateFromNode, ApproximateValueFunction(NNModel))
    getMCTSNNPriorOnly = lambda NNModel, numSimulations: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                                              rollout, backup, selectGreedyAction)
    getMCTSNN = lambda NNModel, numSimulations: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                                     getApproximateValue(NNModel), backup, selectGreedyAction)
    getNN = lambda NNModel, numSimulations: ApproximatePolicy(NNModel, actionSpace)
    getStationaryAgentPolicy = lambda NNModel, numSimulations: stationaryAgentPolicy                                    # should I do this just to keep the interface symmetric?
    getPolicy = GetPolicy(getMCTSNN, getStationaryAgentPolicy)

    # sample trajectory
    getResetFromQPosInit = lambda qPosInit: Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    getResetFromTrial = lambda trial: getResetFromQPosInit(allQPosInit[trial])
    getSampleTrajectory = lambda trial: SampleTrajectory(maxRunningSteps, transit, isTerminal, getResetFromTrial(trial))

    # path to save evaluation trajectories
    trajectoryDirectory = "../../data/trainMCTSNNIteratively/trajectories/evaluate"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, #'modelLearnRate': learningRate,
                                 'modelNumTrialsEachIteration': trainNumTrialsEachIteration, 'numTrials': numTrials,
                                 'trainNumSimulations': trainNumSimulations, 'trainQPosInitNoise': trainQPosInitNoise}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # function to generate trajectories
    restoreNNModelFromIteration = RestoreNNModel(getModelSavePath, generatePolicyNet(hiddenWidths),
                                                              restoreVariables)
    generateTrajectories = GenerateTrajectories(getSampleTrajectory, numTrials, getTrajectorySavePath,
                                                restoreNNModelFromIteration, getPolicy, saveData)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # restoredModelAt25 = restoreNNModelFromIteration(25)
    # NNPrior = ApproximateActionPrior(restoredModelAt25, actionSpace)
    # for trial in range(numTrials):
    #     reset = getResetFromTrial(trial)
    #     state = reset()
    #     NNPrior(state)

    # measurement Function
    initTimeStep = 0
    stateIndex = 0
    getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    getOptimalAction = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepXPos, getWolfXPos, computeAngleBetweenVectors)
    computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction, sheepTransit, getSheepXPos)
    measurementTimeStep = 1
    getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepXPos, getNextStateFromTrajectory)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos, getPosAtNextStepFromTrajectory)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath)
    computeStatistics = ComputeStatistics(loadTrajectories, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()
    numRows = 1
    numColumns = len(manipulatedVariables['numSimulations'])
    plotCounter = 1

    for key, grp in statisticsDf.groupby('numSimulations'):
        grp.index = grp.index.droplevel('numSimulations')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylim(ymin=0, ymax=0.4)
        drawPerformanceLine(grp, axForDraw, key, trainSteps)
        plotCounter += 1

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()