import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
import numpy as np
import pickle
import random
import pygame as pg
from pygame.color import THECOLORS
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy, RandomPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trajectoriesSaveLoad import GetSavePath, loadFromPickle
from src.neuralNetwork.policyNet import GenerateModel, Train, restoreVariables, approximatePolicy, saveVariables
from src.episode import chooseGreedyAction
from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectory


class NeuralNetworkPolicy:
    def __init__(self, model, actionSpace):
        self.model = model
        self.actionSpace = actionSpace

    def __call__(self, state):
        stateFlat = np.asarray(state).flatten()
        graph = self.model.graph
        actionDistribution_ = graph.get_collection_ref("actionDistribution")[0]
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist = self.model.run(actionDistribution_, feed_dict={state_: [stateFlat]})[0]
        actionPrior = {action: prob for action, prob in zip(self.actionSpace, actionDist)}
        return actionPrior


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = [1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index in
                        range(len(self.actionSpace))]

        return oneHotAction


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [pair for trajectory in trajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(lambda pair: pair[self.actionIndex] is not None, stateActionPairs))
        stateActionPairsProcessed = [(np.asarray(state).flatten().tolist(), self.actionToOneHot(action[self.agentId]))
                                     for state, action, actionDist in stateActionPairsFiltered]

        return stateActionPairsProcessed


def main():
    numOfAgent = 2
    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]

    xBoundary = [0, 320]
    yBoundary = [0, 240]
    minDistance = 25

    renderOn = True
    screenColor = THECOLORS['black']
    circleColorList = [THECOLORS['green'], THECOLORS['red']]
    circleSize = 8
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    render = Render(numOfAgent, positionIndex,
                    screen, screenColor, circleColorList, circleSize)

    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)

    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
                       (-8, 0), (-6, -6), (0, -8), (6, -6)]
    heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    randomPolicy = RandomPolicy(actionSpace)
    fixedPolicy = lambda state: {(0, 0): 1}
# mcts
    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    def wolfTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(sheepPolicy(state))])

    # reward function
    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # prior
    getActionPrior = lambda state: {
        action: 1 / len(actionSpace) for action in actionSpace}

    # initialize children; expand
    initializeChildren = InitializeChildren(
        actionSpace, wolfTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolfTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    numSimulations = 10
    mcts = MCTS(numSimulations, selectChild, expand,
                rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    sheepPolicy = fixedPolicy
    wolfPolicy = mcts

    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    # generate trajectories
    maxRunningSteps = 1000
    numTrials = 5
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction, render, renderOn)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/iterativePlayTrainNoPhysicsSheep/trajectories"
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    wolfPolicyName = 'mcts'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'numTrials': numTrials, 'wolfPolicyName': wolfPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()

    # Get dataset for training
    dataSetDirectory = saveDirectory
    dataSetExtension = extension
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = maxRunningSteps
    dataSetNumSimulations = numSimulations
    dataSetNumTrials = numTrials
    dataSetwolfPolicyName = 'mcts'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'wolfPolicyName': dataSetwolfPolicyName}
    dataSetPath = getDataSetPath(dataSetConditionVariables)
    dataSetTrajectories = loadFromPickle(dataSetPath)

    # pre-process the trajectories
    sheepId = 0
    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
    stateActionPairsProcessed = preProcessTrajectories(dataSetTrajectories)

    # shuffle and separate states and actions
    random.shuffle(stateActionPairsProcessed)
    trainData = [[state for state, action in stateActionPairsProcessed],
                 [action for state, action in stateActionPairsProcessed]]

    # initialise model for training
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)
    modelToTrain = generatePolicyNet(hiddenWidths)

    # train models
    allTrainSteps = [0, 10000]
    reportInterval = 100
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    getTrain = lambda trainSteps: Train(trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval,
                                        summaryOn=False, testData=None)

    allTrainFunctions = {trainSteps: getTrain(trainSteps) for trainSteps in allTrainSteps}
    allTrainedModels = {trainSteps: train(generatePolicyNet(hiddenWidths), trainData) for trainSteps, train in
                        allTrainFunctions.items()}

    # get path to save trained models
    fixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps,
                       'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                       'learnRate': learningRate}
    modelSaveDirectory = saveDirectory
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, fixedParameters)
    allModelSavePaths = {trainedModel: getModelSavePath({'trainSteps': trainSteps}) for trainSteps, trainedModel in
                         allTrainedModels.items()}

    # save trained model variables
    savedVariables = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                      allModelSavePaths.items()]


# load chase nn policy
    manipulatedVariables = OrderedDict()
    trainSteps = 10000
    modelSavePath = getModelSavePath({'trainSteps': trainSteps})
    trainedModel = restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath)

    chasePolicy = NeuralNetworkPolicy(trainedModel, actionSpace)


if __name__ == "__main__":
    main()
