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
from src.algorithms.mcts import CalculateScore, SelectChild, InitializeChildren, selectGreedyAction, RollOut, Expand, \
    MCTS, backup
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from exec.evaluationFunctions import GetSavePath, LoadTrajectories, ComputeStatistics
from src.neuralNetwork.policyValueNet import GenerateModelSeparateLastLayer, restoreVariables, ApproximateActionPrior, \
    ApproximateValueFunction, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measurementFunctions import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState, GetAgentPosFromTrajectory, \
    GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def drawPerformanceLine(dataDf, axForDraw, steps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(steps))
        axForDraw.set_ylim([0, 0.4])


class NNValueFunction:
    def __init__(self, trainedModel, getStateFromNode):
        self.trainedModel = trainedModel
        self.getStateFromNode = getStateFromNode

    def __call__(self, leafNode):
        state = self.getStateFromNode(leafNode)
        approximateValueFunction = ApproximateValueFunction(self.trainedModel)
        value = approximateValueFunction(state)

        return value


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

        trainedModel = self.trainedModels[trainSteps]
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations, trainedModel)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)

        if not os.path.isfile(saveFileName):
            allTrajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrials)]
            pickleOut = open(saveFileName, 'wb')
            pickle.dump(allTrajectories, pickleOut)
            pickleOut.close()

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
    manipulatedVariables['trainSteps'] = [10]#[0, 10, 20, 50]
    manipulatedVariables['sheepPolicyName'] = ['NN', 'MCTSNN', 'MCTS']
    manipulatedVariables['numSimulations'] = [1, 2]#[50, 200, 800]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    envModelName = 'twoAgents'
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)                            ###########

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
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getSheepXPos, getWolfXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 12
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    dataSetMaxRunningSteps = 10
    dataSetNumSimulations = 75
    dataSetNumTrials = 1500
    dataSetQPosInit = (0, 0, 0, 0)
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate, 'output': 'policyValue'}
    modelSaveDirectory = "../../data/testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco/trainedModels"
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)
    modelSavePaths = {trainSteps: getModelSavePath({'trainSteps': trainSteps}) for trainSteps in
                      manipulatedVariables['trainSteps']}
    trainedModels = {trainSteps: restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath) for
                     trainSteps, modelSavePath in modelSavePaths.items()}

    # wrapper function for expand
    uniformActionPrior = lambda state: {action: 1/len(actionSpace) for action in actionSpace}
    getExpandUniformPrior = lambda trainedModel: Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit,
                                                                                       uniformActionPrior))
    getExpandNNPrior = lambda trainedModel: Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit,
                                                                                  ApproximateActionPrior(trainedModel,
                                                                                                         actionSpace)))

    # wrapper functions for sheep policies
    getStateFromNode = lambda node: list(node.id.values())[0]
    getMCTS = lambda numSimulations, trainedModel: MCTS(numSimulations, selectChild,
                                                        getExpandUniformPrior(trainedModel), rollout, backup,
                                                        selectGreedyAction)
    getMCTSNN = lambda numSimulations, trainedModel: MCTS(numSimulations, selectChild,
                                                          getExpandNNPrior(trainedModel),
                                                          NNValueFunction(trainedModel, getStateFromNode), backup,
                                                          selectGreedyAction)
    getRandom = lambda numSimulations, trainedModel: lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    getNN = lambda numSimulations, trainedModel: ApproximatePolicy(trainedModel, actionSpace)
    getSheepPolicies = {'MCTS': getMCTS, 'random': getRandom, 'NN': getNN, 'MCTSNN': getMCTSNN}

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # function to generate trajectories
    trajectoryDirectory = "../../data/testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'processing': 'multiCore'}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
                                                getTrajectorySavePath, trainedModels)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

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