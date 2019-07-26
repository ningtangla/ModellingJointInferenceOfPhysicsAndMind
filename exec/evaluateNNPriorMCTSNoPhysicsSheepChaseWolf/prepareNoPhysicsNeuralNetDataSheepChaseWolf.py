import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
import numpy as np
import pickle
import random
import pygame as pg
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyNet import GenerateModel, Train, restoreVariables, approximatePolicy
from src.episode import chooseGreedyAction
from exec.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectory


class SampleTrajectorySingleAgentActions:
    def __init__(self, sampleTrajectory, agentId):
        self.sampleTrajectory = sampleTrajectory
        self.agentId = agentId

    def __call__(self, policy):
        trajectory = self.sampleTrajectory(policy)
        trajectorySingleAgentActions = [(state, action[self.agentId]) if action is not None else (state, None)
                                        for state, action in trajectory]

        return trajectorySingleAgentActions


def generateData(sampleTrajectory, policy, actionSpace, trajNumber, path):
    totalStateBatch = []
    totalActionBatch = []
    for index in range(trajNumber):
        if index % 100 == 0:
            print(index)
        trajectory = sampleTrajectory(policy)
        states, actions = zip(*trajectory)
        totalStateBatch = totalStateBatch + list(states)
        oneHotActions = [[1 if (np.array(action) == np.array(actionSpace[index])).all() else 0 for index in
                          range(len(actionSpace))] for action in actions]
        totalActionBatch = totalActionBatch + oneHotActions

    dataSet = list(zip(totalStateBatch, totalActionBatch))
    saveFile = open(path, "wb")
    pickle.dump(dataSet, saveFile)


def loadData(path):
    pklFile = open(path, "rb")
    dataSet = pickle.load(pklFile)
    pklFile.close()
    return dataSet


def sampleData(data, batchSize):
    batch = random.sample(data, batchSize)
    batchInput = [x for x, _ in batch]
    batchOutput = [y for _, y in batch]
    return batchInput, batchOutput


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


def main():
    numOfAgent = 2
    sheepId = 0
    wolfId = 1
    positionIndex = [0, 1]

    xBoundary = [0, 320]
    yBoundary = [0, 240]
    minDistance = 25

    renderOn = True
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
    isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, minDistance)
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
    #                    (-8, 0), (-6, -6), (0, -8), (6, -6)]

    # wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(
    #     wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    # load trained escaped NN
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]  # [64]*3
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    model = generatePolicyNet(hiddenWidths)
    manipulatedVariables = OrderedDict()
    dataSetMaxRunningSteps = 30
    dataSetNumSimulations = 200
    dataSetNumTrials = 2
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate}
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
    modelSaveExtension = ''
    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)

    trainSteps = 200
    modelSavePath = getModelSavePath({'trainSteps': trainSteps})
    trainedModel = restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath)

    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
                       (-8, 0), (-6, -6), (0, -8), (6, -6)]
    escapePolicy = NeuralNetworkPolicy(trainedModel, wolfActionSpace)

    def sheepTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(escapePolicy(state))])

    # reward function
    maxRolloutSteps = 10

    aliveBonus = - 1 / maxRolloutSteps
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # initialize children; expand
    initializeChildren = InitializeChildren(
        actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorPos, getPreyPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    numSimulations = 200
    mcts = MCTS(numSimulations, selectChild, expand,
                rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    def policy(state): return [mcts(state), escapePolicy(state)]

    # generate trajectories
    maxRunningSteps = 30
    numTrials = 5000
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction, render, renderOn)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/evaluateNNPriorMCTSNoPhysicsSheepChaseWolf/trajectories"
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)
    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
    sheepPolicyName = 'mcts'
    conditionVariables = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'numTrials': numTrials, 'sheepPolicyName': sheepPolicyName}
    path = getSavePath(conditionVariables)

    pickleIn = open(path, 'wb')
    pickle.dump(trajectories, pickleIn)
    pickleIn.close()


if __name__ == "__main__":
    main()
