import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import tensorflow as tf
import random
from functools import reduce
import pickle

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import CalculateScore, SelectChild, InitializeChildren, selectGreedyAction, Expand, MCTS, backup
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNet import GenerateModelSeparateLastLayer, ApproximateActionPrior, \
    ApproximateValueFunction, Train, saveVariables
from src.constrainedChasingEscapingEnv.wrapperFunctions import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientController, TrainTerminalController, TrainReporter
from exec.trainMCTSNNIteratively.wrappers import GetApproximateValueFromNode, getStateFromNode


def saveData(data, path):
    pickleOut = open(path, 'wb')
    pickle.dump(data, pickleOut)
    pickleOut.close()


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index
                                   in range(len(self.actionSpace))])
        return oneHotAction


class AccumulateRewards:
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([reduce(accumulateReward, reversed(rewards[TimeT:])) for TimeT in range(len(rewards))])

        return accumulatedRewards


class AddValuesToTrajectory:
    def __init__(self, trajectoryValueFunction):
        self.trajectoryValueFunction = trajectoryValueFunction

    def __call__(self, trajectory):
        values = self.trajectoryValueFunction(trajectory)
        trajWithValues = [(s, a, np.array([v])) for (s, a), v in zip(trajectory, values)]

        return trajWithValues


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot, addValuesToTrajectory):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot
        self.addValuesToTrajectory = addValuesToTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        stateActionValueTriples = [triple for trajectory in trajectoriesWithValues for triple in trajectory]
        triplesFiltered = list(filter(lambda triple: triple[self.actionIndex] is not None, stateActionValueTriples))
        processTriple = lambda state, actions, value: (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        triplesProcessed = [processTriple(state, actions, value) for state, actions, value in triplesFiltered]

        random.shuffle(triplesProcessed)
        trainData = [[state for state, action, value in triplesProcessed],
                     [action for state, action, value in triplesProcessed],
                     np.asarray([value for state, action, value in triplesProcessed])]

        return trainData


class GetPolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, NNModel):
        sheepPolicy = self.getSheepPolicy(NNModel)
        wolfPolicy = self.getWolfPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class PlayToTrain:
    def __init__(self, sampleTrajectory, numTrialsEachIteration, getTrajectorySavePath, getPolicy, saveData):
        self.sampleTrajectory = sampleTrajectory
        self.numTrialsEachIteration = numTrialsEachIteration
        self.getTrajectorySavePath = getTrajectorySavePath
        self.getPolicy = getPolicy
        self.saveData = saveData

    def __call__(self, NNModel, pathParameters):
        policy = self.getPolicy(NNModel)
        trajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrialsEachIteration)]
        trajectorySavePath = self.getTrajectorySavePath(pathParameters)
        self.saveData(trajectories, trajectorySavePath)

        return trajectories


class TrainToPlay:
    def __init__(self, train, getModelSavePath):
        self.train = train
        self.getModelSavePath = getModelSavePath

    def __call__(self, NNModel, trainData, pathParameters):
        updatedNNModel = self.train(NNModel, trainData)
        modelSavePath = self.getModelSavePath(pathParameters)
        saveVariables(updatedNNModel, modelSavePath)

        return updatedNNModel


def main():
    random.seed(128)
    np.random.seed(128)
    tf.set_random_seed(128)

    # commonly varied parameters
    numIterations = 1000
    numTrialsEachIteration = 1

    # functions for MCTS
    envModelName = 'twoAgents'
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
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

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network model
    numStateSpace = 12
    learningRates = [1e-8, 1e-6, 1e-10]
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]

    maxRunningSteps = 10
    numSimulations = 75                                                                                                 # should we change this number?

    # wrapper function for expand
    approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, sheepTransit, approximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))

    # wrapper function for policy
    getApproximateValue = lambda NNModel: GetApproximateValueFromNode(getStateFromNode, ApproximateValueFunction(NNModel))
    getMCTSNN = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                     getApproximateValue(NNModel), backup, selectGreedyAction)
    getStationaryAgentPolicy = lambda NNModel: stationaryAgentPolicy                                                    # should I do this just to keep the interface symmetric?
    getPolicy = GetPolicy(getMCTSNN, getStationaryAgentPolicy)

    # sample trajectory
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    # pre-process the trajectory for training the neural network
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot, addValuesToTrajectory)

    # function to train NN model
    trainSteps = 50
    batchSize = None
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientController(initActionCoeff, initValueCoeff)
    reportInterval = 25
    trainReporter = TrainReporter(trainSteps, reportInterval)

    train = Train(trainSteps, batchSize, terminalController, coefficientController, trainReporter)

    # NN model save path
    trainFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations, 'trainSteps': trainSteps,
                            'numTrialsEachIteration': numTrialsEachIteration}
    NNModelSaveDirectory = "../../data/trainMCTSNNIteratively/trainedNNModels"
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)

    # trajectory save path
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations,
                                 'trainSteps': trainSteps,
                                 'numTrialsEachIteration': numTrialsEachIteration}
    trajectorySaveDirectory = "../../data/trainMCTSNNIteratively/trajectories/training"
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # play and train NN iteratively
    playToTrain = PlayToTrain(sampleTrajectory, numTrialsEachIteration, getTrajectorySavePath, getPolicy, saveData)
    trainToPlay = TrainToPlay(train, getModelSavePath)
    replayBuffer = []


    for learningRate in learningRates:
        generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate,
                                                           regularizationFactor)
        NNModel = generatePolicyNet(hiddenWidths)
        for iteration in range(numIterations):
            trajectories = playToTrain(NNModel, {'iteration': iteration, 'learnRate': learningRate})
            trainData = preProcessTrajectories(trajectories)

            updatedNNModel = trainToPlay(NNModel, trainData, {'iteration': iteration, 'learnRate': learningRate})
            NNModel = updatedNNModel


if __name__ == '__main__':
    main()