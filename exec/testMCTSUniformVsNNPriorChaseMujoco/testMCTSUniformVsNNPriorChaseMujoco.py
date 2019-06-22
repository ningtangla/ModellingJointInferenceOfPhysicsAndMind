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


class NNSampleTrajectory:
    def __init__(self, transit, neuralNetworkPolicy):
        self.transit = transit
        self.neuralNetworkPolicy = neuralNetworkPolicy

    def __call__(self, initState):
        initAllAgentActions = self.neuralNetworkPolicy(initState)
        newState = self.transit(initState, initAllAgentActions)
        newAllAgentActions = self.neuralNetworkPolicy(newState)

        trajectory = [(initState, initAllAgentActions), (newState, newAllAgentActions)]
        return trajectory


class NNGenerateTrajectories:
    def __init__(self, getSavePath, nnSampleTrajectory):
        self.getSavePath = getSavePath
        self.nnSampleTrajectory = nnSampleTrajectory

    def __call__(self, parameters):
        parameters['sheepPolicyName'] = 'MCTSNN'
        MCTSNNTrajPath = self.getSavePath(parameters)
        pickleIn = open(MCTSNNTrajPath, 'rb')
        MCTSNNTrajectories = pickle.load(pickleIn)
        allMCTSNNInitStates = [trajectory[0][0] for trajectory in MCTSNNTrajectories]

        NNTrajectories = [self.nnSampleTrajectory(initState) for initState in allMCTSNNInitStates]

        return NNTrajectories


def drawPerformanceLine(dataDf, axForDraw, steps):
    for key, grp in dataDf.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        # grp.plot(ax=axForDraw, label=key, y='mean', yerr='std', title='TrainSteps: {}'.format(steps))
        grp.plot(ax=axForDraw, label=key, y='mean', title='TrainSteps: {}'.format(steps))
        axForDraw.set_ylim([0, 0.4])


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
    def __init__(self, getSampleTrajectory, getSheepPolicies, wolfPolicy, numTrials, getSavePath, trainedModels, transit):
        self.getSampleTrajectory = getSampleTrajectory
        self.getSheepPolicies = getSheepPolicies
        self.wolfPolicy = wolfPolicy
        self.numTrials = numTrials
        self.getSavePath = getSavePath
        self.trainedModels = trainedModels
        self.transit = transit

    def __call__(self, oneConditionDf):
        startTime = time.time()
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        maxInitDistance = oneConditionDf.index.get_level_values('maxInitDistance')[0]

        sampleTrajectory = self.getSampleTrajectory(maxInitDistance)

        trainedModel = self.trainedModels[trainSteps]
        getSheepPolicy = self.getSheepPolicies[sheepPolicyName]
        sheepPolicy = getSheepPolicy(numSimulations, trainedModel)
        policy = lambda state: [sheepPolicy(state), self.wolfPolicy(state)]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)

        if sheepPolicyName == 'NN':
            if os.path.isfile(saveFileName):
                print("NN file existed")
            nnSampleTrajectory = NNSampleTrajectory(self.transit, policy)
            nnGenerateTrajectories = NNGenerateTrajectories(self.getSavePath, nnSampleTrajectory)
            allNNTrajectories = nnGenerateTrajectories(parameters)
            pickleOut = open(saveFileName, 'wb')
            pickle.dump(allNNTrajectories, pickleOut)
            pickleOut.close()

        if not os.path.isfile(saveFileName):
            allTrajectories = [sampleTrajectory(policy) for trial in range(self.numTrials)]
            pickleOut = open(saveFileName, 'wb')
            pickle.dump(allTrajectories, pickleOut)
            pickleOut.close()

        endTime = time.time()
        print("Time for policy {}, numSimulations {}, trainSteps {} = {}".format(sheepPolicyName, numSimulations,
                                                                                 trainSteps, (endTime-startTime)))

        return None


class Check:
    def __call__(self, oneConditionDf):
        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}

        if parameters['sheepPolicyName'] == 'NN':
            filePath = getSavePath(parameters)
            pickleIn = open(filePath, 'rb')
            NNTrajectories = pickle.load(pickleIn)
            NNinitStates = [trajectory[0] for trajectory in NNTrajectories]
            pickleIn.close()

            MCTSFilePath = filePath.replace('sheepPolicyName=NN', 'sheepPolicyName=MCTSNN')
            pickleIn = open(MCTSFilePath, 'rb')
            MCTSNNTrajectories = pickle.load(pickleIn)
            MCTSNNinitStates = [trajectory[0] for trajectory in MCTSNNTrajectories]
            pickleIn.close()

            truthValue = NNinitStates == MCTSNNinitStates
            print(truthValue)


def main():
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # manipulated variables (and some other parameters that are commonly varied)
    numTrials = 50
    maxRunningSteps = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['maxInitDistance'] = [30.0]#[2.5, 30]
    manipulatedVariables['trainSteps'] = [0, 20, 50]#[0, 50, 100, 500]
    manipulatedVariables['sheepPolicyName'] = ['NN', 'MCTSNN']
    manipulatedVariables['numSimulations'] = [50, 200, 800]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # functions for MCTS
    qPosInit = (0, 0, 0, 0)
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
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

    # sample trajectory
    # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    getSampleTrajectory = lambda maxInitDistance: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset,
                                                                   maxInitDistance)

    # function to generate trajectories
    trajectoryDirectory = "../../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories"
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    fixedParameters = {'numTrials': numTrials, 'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, fixedParameters)

    # generateTrajectories = GenerateTrajectories(sampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
    #                                             getTrajectorySavePath, trainedModels)
    generateTrajectories = GenerateTrajectories(getSampleTrajectory, getSheepPolicies, stationaryAgentPolicy, numTrials,
                                                getTrajectorySavePath, trainedModels, transit)


    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # check if NN and MCTSNN has same starting initState
    toSplitFrame.groupby(levelNames).apply(check)

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

    combinedMeanDf = statisticsDf.groupby(['trainSteps', 'sheepPolicyName', 'numSimulations']).agg('mean')

    print('statisticsDf')
    print(statisticsDf)
    print('combinedMeanDf')
    print(combinedMeanDf)

    # plot the statistics
    fig = plt.figure()

    numColumns = len(manipulatedVariables['trainSteps'])
    # numRows = len(manipulatedVariables['maxInitDistance'])
    numRows = 1
    plotCounter = 1

    for steps, grpSteps in combinedMeanDf.groupby('trainSteps'):
        grpSteps.index = grpSteps.index.droplevel('trainSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        drawPerformanceLine(grpSteps, axForDraw, steps)
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()