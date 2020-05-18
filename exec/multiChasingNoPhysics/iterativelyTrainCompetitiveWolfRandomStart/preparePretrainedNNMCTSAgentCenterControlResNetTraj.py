import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateState
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, StochasticMCTS, backup, establishPlainActionDistFromMultipleTrees, Expand
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import Render, SampleTrajectoryWithRender, SampleAction, chooseGreedyAction, SelectSoftmaxAction
from exec.parallelComputing import GenerateTrajectoriesParallel


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, fixNNAgentId, fixedNNPolicy, chooseAction):
        self.fixNNAgentId = fixNNAgentId
        self.fixedNNPolicy = fixedNNPolicy
        self.chooseAction = chooseAction

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicy]
        multiAgentActions.insert(agentId, selfAction)

        fixedNNAgentAction = self.chooseAction(self.fixedNNPolicy(state))
        multiAgentActions.insert(self.fixNNAgentId, fixedNNAgentAction)

        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numTree, numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numTree = numTree
        self.numSimulations = numSimulations
        self.numSimulationsPerTree = int(self.numSimulations / self.numTree)
        self.actionSpaceList = actionSpaceList
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy[agentId](selfNNModel)

        def transitInMCTS(state, selfAction): return self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaceList[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = StochasticMCTS(self.numTree, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([approximatePolicy(NNModel) for approximatePolicy, NNModel in zip(self.approximatePolicy, multiAgentNNModel)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
                                     for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy

        def policy(state): return [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy


def main():
    DEBUG = 0
    renderOn = 0

    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 10
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
        iterationIndex = 1
        numTrainStepEachIteration = 1
        numTrajectoriesPerIteration = 1

    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
        iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])
        numTrainStepEachIteration = int(parametersForTrajectoryPath['numTrainStepEachIteration'])
        numTrajectoriesPerIteration = int(parametersForTrajectoryPath['numTrajectoriesPerIteration'])

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiChasingNoPhysics', 'iterativelyTrainCompetitiveWolf', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'

    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 50
    numTree = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        # No physics env
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        posIndex = [0, 1]

        getSheepXPos = GetAgentPosFromState(sheepId, posIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, posIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, posIndex)

        numOfAgent = 3
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)

        def isTerminal(state): return isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        numFramesToInterpolate = 3
        transit = TransitWithInterpolateState(numFramesToInterpolate, transitionFunction, isTerminal)

        # action
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        actionSpaceList = [sheepActionSpace, wolfActionOneSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolfActionOneSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolfOneModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        generateModelList = [generateSheepModel, generateWolfOneModel, generateWolfOneModel]

        sheepDepth = 9
        wolfDepth = 9
        depthList = [sheepDepth, wolfDepth, wolfDepth]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        trainableAgentIds = [sheepId, wolfOneId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

        otherAgentApproximatePolicy = [lambda NNmodel, : ApproximatePolicy(NNmodel, sheepActionSpace), lambda NNmodel, : ApproximatePolicy(NNmodel, wolfActionOneSpace)]
        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getApproximatePolicy = [lambda NNmodel, : ApproximatePolicy(NNmodel, sheepActionSpace), lambda NNmodel, : ApproximatePolicy(NNmodel, wolfActionOneSpace)]
        getApproximateValue = [lambda NNmodel: ApproximateValue(NNmodel), lambda NNmodel: ApproximateValue(NNmodel)]

        fixNNAgentId = wolfTwoId
        fixedNNPolicy = ApproximatePolicy(multiAgentNNmodel[wolfOneId], wolfActionOneSpace)
        fixedWolfNNPolicy = lambda state: fixedNNPolicy([state[0], state[2], state[1]])

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)

        def getStateFromNode(node): return list(node.id.values())[0]

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(fixNNAgentId, fixedWolfNNPolicy, chooseActionInMCTS)
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numTree, numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)
        mctsPolicy = prepareMultiAgentPolicy(multiAgentNNmodel)

        def policy(state):
            pilcyList = list(mctsPolicy(state))
            pilcyList.insert(fixNNAgentId, fixedWolfNNPolicy(state))
            return pilcyList

        # sample and save trajectories
        chooseActionList = [chooseGreedyAction, chooseGreedyAction, chooseGreedyAction]

        render = None
        if renderOn:
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['orange']]
            circleSize = 10
            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)
            screen = pg.display.set_mode([max(xBoundary), max(yBoundary)])
            render = Render(numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
