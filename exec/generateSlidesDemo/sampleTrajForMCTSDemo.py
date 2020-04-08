import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
import math
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
import pygame as pg
from pygame.color import THECOLORS
from anytree import AnyNode as Node

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishPlainActionDistFromMultipleTrees
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import Render, SampleTrajectoryWithRender, SampleAction, chooseGreedyAction, SelectSoftmaxAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.generateSlidesDemo.MCTSVisualization import ScalePos, InitializeScreen, DrawBackground, DrawState, MCTSRender


class MCTS:
    def __init__(self, numSimulation, selectChild, expand, estimateValue, backup, outputDistribution, mctsRender, mctsRenderOn):
        self.numSimulation = numSimulation
        self.selectChild = selectChild
        self.expand = expand
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputDistribution = outputDistribution
        self.mctsRender = mctsRender
        self.mctsRenderOn = mctsRenderOn

    def __call__(self, currentState):
        roots = []
        backgroundScreen = None
        root = Node(id={None: currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                nextNode = self.selectChild(currentNode)
                if self.mctsRenderOn:
                    backgroundScreen = self.mctsRender(currentNode, nextNode, roots, backgroundScreen)
                nodePath.append(nextNode)
                currentNode = nextNode

            leafNode = self.expand(currentNode)
            value = self.estimateValue(leafNode)
            self.backup(value, nodePath)
        roots.append(root)
        actionDistribution = self.outputDistribution(roots)
        return actionDistribution


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction, reasonMindFunctions):
        self.chooseAction = chooseAction
        self.reasonMindFunctions = reasonMindFunctions

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        reasonMind = list(self.reasonMindFunctions[agentId])
        del reasonMind[agentId]
        othersPolicyInMCTS = [reason(policy) for reason, policy in zip(reasonMind, othersPolicy)]
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicyInMCTS]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentMCTS():
    def __init__(self, numSimulations, actionSpaces, maxRolloutSteps, rewardFunctions, rolloutHeuristicFunctions, selectChild, isTerminal, transit, getApproximateActionPrior, composeMultiAgentTransitInSingleAgentMCTS, MCTSRenders, MCTSRenderOn):
        self.numSimulations = numSimulations
        self.actionSpaces = actionSpaces
        self.maxRolloutSteps = maxRolloutSteps
        self.rewardFunctions = rewardFunctions
        self.rolloutHeuristicFunctions = rolloutHeuristicFunctions
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getApproximateActionPrior = getApproximateActionPrior
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS
        self.MCTSRenders = MCTSRenders
        self.MCTSRenderOn = MCTSRenderOn

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximateActionPrior(selfNNModel, self.actionSpaces[agentId])
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaces[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        rolloutPolicy = lambda state: self.actionSpaces[agentId][np.random.choice(range(len(self.actionSpaces[agentId])))]
        rewardFunction = self.rewardFunctions[agentId]
        rolloutHeuristic = self.rolloutHeuristicFunctions[agentId]
        estimateValue = RollOut(rolloutPolicy, self.maxRolloutSteps, transitInMCTS, rewardFunction, self.isTerminal, rolloutHeuristic)

        MCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees,
                          self.MCTSRenders[agentId], self.MCTSRenderOn)

        return MCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, MCTSAgentIds, actionSpaces, composeSingleAgentPolicy, getApproximatePolicy, MCTSRenderInterval, callTime):
        self.MCTSAgentIds = MCTSAgentIds
        self.actionSpaces = actionSpaces
        self.composeSingleAgentPolicy = composeSingleAgentPolicy
        self.getApproximatePolicy = getApproximatePolicy
        self.MCTSRenderInterval = MCTSRenderInterval
        self.callTime = callTime

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([self.getApproximatePolicy(NNModel, actionSpace) for NNModel, actionSpace in zip(multiAgentNNModel, self.actionSpaces)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)

        if self.callTime % self.MCTSRenderInterval == 0:
            self.composeSingleAgentPolicy.MCTSRenderOn = True
        else:
            self.composeSingleAgentPolicy.MCTSRenderOn = False
        print(self.callTime)
        print(self.composeSingleAgentPolicy.MCTSRenderOn)
        self.callTime += 1

        MCTSAgentsPolicy = np.array([self.composeSingleAgentPolicy(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy) for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)

        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy


class SampleTrajectoryForMCTSDemo:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        return trajectory


class DrawState:
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround

    def __call__(self, state, circleColorList):
        self.drawBackGround()
        for agentIndex in range(self.numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        pg.display.flip()
        return self.screen


class MCTSRender():
    def __init__(self, numAgent, MCTSAgentId, screen, surfaceWidth, surfaceHeight, screenColor, circleColorList, mctsLineColor, circleSize, saveImage, saveImageDir, drawState, scalePos):
        self.numAgent = numAgent
        self.MCTSAgentId = MCTSAgentId
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.mctsLineColor = mctsLineColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir
        self.drawState = drawState
        self.scalePos = scalePos

    def __call__(self, currNode, nextNode, roots, backgroundScreen):
        parentNumVisit = currNode.numVisited
        parentValueToTal = currNode.sumValue
        originalState = list(currNode.id.values())[0]
        poses = self.scalePos(originalState)

        childNumVisit = nextNode.numVisited
        childValueToTal = nextNode.sumValue
        originalNextState = list(nextNode.id.values())[0]
        nextPoses = self.scalePos(originalNextState)

        if not os.path.exists(self.saveImageDir):
            os.makedirs(self.saveImageDir)

        if len(roots) > 0 and nextNode.depth == 1:
            nodeIndex = currNode.children.index(nextNode)
            grandchildren_visit = np.sum([[child.numVisited for child in anytree.findall(root, lambda node: node.depth == 1)] for root in roots], axis=0)
            lineWidth = math.ceil(0.3 * (grandchildren_visit[nodeIndex] + 1))
        else:
            lineWidth = math.ceil(0.3 * (nextNode.numVisited + 1))

        surfaceToDraw = pg.Surface((self.surfaceWidth, self.surfaceHeight))
        surfaceToDraw.fill(self.screenColor)
        if backgroundScreen == None:
            backgroundScreen = self.drawState(poses, self.circleColorList)
            if self.saveImage == True:
                for numStaticImage in range(120):
                    filenameList = os.listdir(self.saveImageDir)
                    pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')

        surfaceToDraw.set_alpha(180)
        surfaceToDraw.blit(backgroundScreen, (0, 0))
        self.screen.blit(surfaceToDraw, (0, 0))

        pg.display.flip()
        pg.time.wait(1)

        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit

            for i in range(self.numAgent):
                oneAgentPosition = np.array(poses[i])
                oneAgentNextPosition = np.array(nextPoses[i])
                if i != 0:  # draw mcts line for wolves
                    pg.draw.line(surfaceToDraw, self.mctsLineColor, [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], [np.int(oneAgentNextPosition[0]), np.int(oneAgentNextPosition[1])], lineWidth)
                pg.draw.circle(surfaceToDraw, self.circleColorList[i], [np.int(oneAgentNextPosition[0]), np.int(oneAgentNextPosition[1])], self.circleSize)

            self.screen.blit(surfaceToDraw, (0, 0))
            pg.display.flip()
            pg.time.wait(1)

            if self.saveImage == True:
                filenameList = os.listdir(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')
        return self.screen


def main():
    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    parametersForTrajectoryPath = {}
    startSampleIndex = 0
    endSampleIndex = 1
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'generateMCTSDemo', 'image')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    # get traj save path
    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 100
    numSimulations = 100
    killzoneRadius = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    mctsDemoParameters = parametersForTrajectoryPath.copy()
    mctsDemoParameters.update({'index': startSampleIndex})
    mctsDemoSavePath = generateTrajectorySavePath(mctsDemoParameters)

    if not os.path.isfile(trajectorySavePath):
        # No physics env
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        numAgent = 2
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
        isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

        wolvesId = 1
        centerControlIndexList = [wolvesId]
        unpackAction = UnpackCenterControlAction(centerControlIndexList)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity, unpackAction)

        # product wolves action space
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 9
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 6
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))
        actionSpaceList = [sheepActionSpace, wolvesActionSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        generateModelList = [generateSheepModel, generateWolvesModel]

        sheepDepth = 5
        wolfDepth = 9
        depthList = [sheepDepth, wolfDepth]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        trainableAgentIds = [sheepId, wolvesId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

        # restorePretrainModel
        sheepPreTrainModelPath = os.path.join(dirName, '..', '..', 'data', 'preTrainModel', 'agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        wolvesPreTrainModelPath = os.path.join(dirName, '..', '..', 'data', 'preTrainModel', 'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        pretrainModelPathList = [sheepPreTrainModelPath, wolvesPreTrainModelPath]

        for agentId in trainableAgentIds:
            restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], pretrainModelPathList[agentId])
            multiAgentNNmodel[agentId] = restoredNNModel

        # otherAgentApproximatePolicy = [lambda NNmodel, : ApproximatePolicy(NNmodel, sheepActionSpace), lambda NNmodel, : ApproximatePolicy(NNmodel, wolvesActionSpace)]

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getApproximatePolicy = lambda NNmodel, actionSpace: ApproximatePolicy(NNmodel, actionSpace)
        getApproximateUniformActionPrior = lambda NNModel, actionSpace: lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

        # sample and save trajectories
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]

        saveImage = False
        saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        betaInMCTS = 1
        chooseActionInMCTS = SampleAction(betaInMCTS)

        reasonMindList = np.array([[lambda policy: RandomPolicy(actionSpace) for actionSpace in actionSpaceList] for subjectiveAgentId in range(numAgent)])

        reasonMindList[sheepId][wolvesId] = lambda policy: policy
        reasonMindList[wolvesId][sheepId] = lambda policy: policy

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS, reasonMindList)

        screenWidth = 800
        screenHeight = 800
        fullScreen = False
        initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
        screen = initializeScreen()
        leaveEdgeSpace = 195
        lineWidth = 4
        circleSize = 10
        xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
        yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]
        screenColor = THECOLORS['black']
        lineColor = THECOLORS['white']

        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
        drawStateWithRope = DrawState(screen, circleSize, numOfAgent, posIndex, drawBackground)

        rawXRange = [0, 800]
        rawYRange = [0, 800]
        scaledXRange = [200, 600]
        scaledYRange = [200, 600]
        scalePos = ScalePos(posIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

        circleColorList = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]

        mctsLineColor = np.array([240, 240, 240, 180])
        circleSizeForMCTS = int(0.6 * circleSize)
        saveImage = False
        saveImageDir = os.path.join(mctsDemoSavePath, "image")
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        mctsRenders = [MCTSRender(numOfAgent, MCTSAgentId, screen, screenWidth, screenHeight, screenColor, circleColorList, mctsLineColor, circleSizeForMCTS, saveImage, saveImageDir, drawStateWithRope, scalePos) for MCTSAgentId in range(numAgent)]
        mctsRenderOn = True

        maxRolloutSteps = 10
        rewardFunctions = [RewardFunctionCompete(-terminalReward, terminalReward, isTerminal) for terminalReward in terminalRewardList]

        def rolloutHeuristic(rolloutHeuristicWeight):
            rolloutHeuristic1 = HeuristicDistanceToTarget(
                rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos)
            rolloutHeuristic2 = HeuristicDistanceToTarget(
                rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos)
            return lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2

        rolloutHeuristicWeight = [-0.1, 0.1]
        rolloutHeuristics = [rolloutHeuristic(weight) for weight in rolloutHeuristicWeight]
        composeSingleAgentMCTS = ComposeSingleAgentMCTS(numSimulations, actionSpaceList, maxRolloutSteps, rewardFunctions, rolloutHeuristics, selectChild, isTerminal, transit, getApproximatePolicy, composeMultiAgentTransitInSingleAgentMCTS, mctsRenders, mctsRenderOn)

        MCTSAgentIds = [wolvesId]
        MCTSRenderInterval = 5
        callTime = 0
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(MCTSAgentIds, actionSpaceList, composeSingleAgentMCTS, getApproximatePolicy, MCTSRenderInterval, callTime)

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        beta = 1
        selectSoftmaxAction = SelectSoftmaxAction(beta)
        chooseActionList = [chooseGreedyAction, selectSoftmaxAction]
        sampleTrajectory = SampleTrajectoryForMCTSDemo(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

        numTrials = 10
        trajectories = [sampleTrajectory(policy) for _ in range(numTrials)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()

# ffmpeg -r 120 -f image2 -s 1920x1080 -i  %0d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/generateMCTSDemo/demo.mp4
