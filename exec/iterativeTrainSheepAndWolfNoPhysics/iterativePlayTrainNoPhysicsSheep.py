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

    wolfActionSpace = [(6, 0), (5, 5), (0, 6), (-5, 5),
                       (-10, 0), (-5, -5), (0, -10), (5, -5)]
    heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

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
    def policy(state): return [mcts(state), heatSeekingPolicy(state)]

    # generate trajectories
    maxRunningSteps = 30
    numTrials = 5000
    sampleTrajectory = SampleTrajectory(
        maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction, render, renderOn)
    trajectories = [sampleTrajectory(policy) for trial in range(numTrials)]

    # save the trajectories
    saveDirectory = "../../data/iterativePlayTrainNoPhysicsSheep/trajectories"
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

    # Get dataset for training
    dataSetDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trajectories"
    dataSetExtension = '.pickle'
    getDataSetPath = GetSavePath(dataSetDirectory, dataSetExtension)
    dataSetMaxRunningSteps = 30
    dataSetNumSimulations = 200
    dataSetNumTrials = 2
    dataSetSheepPolicyName = 'mcts'
    dataSetConditionVariables = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'sheepPolicyName': dataSetSheepPolicyName}
    dataSetPath = getDataSetPath(dataSetConditionVariables)
    dataSetTrajectories = loadData(dataSetPath)

    # pre-process the trajectories
    sheepId = 0
    actionIndex = 1
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
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
    allTrainSteps = [0, 100, 500]
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
    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysics/trainedModels"
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
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    model = generatePolicyNet(hiddenWidths)
    manipulatedVariables = OrderedDict()
    dataSetMaxRunningSteps = 30
    dataSetNumSimulations = 200
    dataSetNumTrials = 3
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate}
    numStateSpace = 4
    numActionSpace = len(actionSpace)
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    modelSaveDirectory = "../../data/evaluateNNPriorMCTSNoPhysicsSheepChaseWolf/trainedModels"
    modelSaveExtension = ''
    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)

    trainSteps = 200
    modelSavePath = getModelSavePath({'trainSteps': trainSteps})
    trainedModel = restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath)

    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
                       (-8, 0), (-6, -6), (0, -8), (6, -6)]

    chasePolicy = NeuralNetworkPolicy(trainedModel, wolfActionSpace)


if __name__ == "__main__":
    main()
