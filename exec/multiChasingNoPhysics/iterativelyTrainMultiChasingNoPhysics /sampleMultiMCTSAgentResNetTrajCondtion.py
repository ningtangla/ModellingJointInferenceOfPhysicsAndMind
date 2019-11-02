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


# from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform

from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysics, Reset,IsTerminal,StayInBoundaryByReflectVelocity

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
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist,Expand
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction):
        self.chooseAction = chooseAction

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicy]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numSimulations = numSimulations
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
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaceList[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([approximatePolicy(NNModel) for approximatePolicy, NNModel in zip( self.approximatePolicy,multiAgentNNModel)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
                                     for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy


def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 150
    numSimulations = 100
    killzoneRadius = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])

    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # parametersForTrajectoryPath={}
    # startSampleIndex=0
    # endSampleIndex=10
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # parametersForTrajectoryPath['iterationIndex']=0

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        numOfAgent=3
        agentIds = list(range(numOfAgent))
        
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]
        xBoundary = [0,600]
        yBoundary = [0,600]

        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos =GetAgentPosFromState(wolfTwoId, xPosIndex)


        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepAliveBonus = 1/maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward,wolfTerminalReward]

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
        isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary) 
        transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        actionSpaceList=[sheepActionSpace,wolfActionOneSpace,wolfActionTwoSpace]
        # getApproximatePolicy = lambda NNmodel,: ApproximatePolicy(NNmodel, actionSpace)

        getApproximatePolicy =[lambda NNmodel,: ApproximatePolicy(NNmodel, sheepActionSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolfActionOneSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolfActionTwoSpace)]

        getApproximateValue = [lambda NNmodel: ApproximateValue(NNmodel),lambda NNmodel: ApproximateValue(NNmodel),lambda NNmodel: ApproximateValue(NNmodel)]

        getStateFromNode = lambda node: list(node.id.values())[0]

        # neural network init
        numStateSpace = 6
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsTwoWolves', 'NNModelRes')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    
        startTime = time.time()
        trainableAgentIds = [sheepId, wolfOneId,wolfTwoId]

        depth = 5
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in agentIds]

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)
        chooseActionList = [chooseActionInMCTS, chooseActionInMCTS,chooseActionInMCTS]
        # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

        from exec.evaluateNoPhysicsEnvWithRender import Render #SampleTrajectoryWithRender
        import pygame as pg
        renderOn = False
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        saveImage = False
        saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        render = Render(numOfAgent, xPosIndex,
                        screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        otherAgentApproximatePolicy = [lambda NNmodel,: ApproximatePolicy(NNmodel, sheepActionSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolfActionOneSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolfActionTwoSpace)]

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS)
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)


        iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])

        for agentId in trainableAgentIds:
            if iterationIndex in [0,1] :
                modelPath = generateNNModelSavePath({'iterationIndex': 0, 'agentId': agentId})
            else:
                numTrainStepEachIteration = int(parametersForTrajectoryPath['numTrainStepEachIteration'])
                numTrajectoriesPerIteration = int(parametersForTrajectoryPath['numTrajectoriesPerIteration'])

                modelPath = generateNNModelSavePath({'iterationIndex': iterationIndex-1, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
            restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPath)
            multiAgentNNmodel[agentId] = restoredNNModel

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)

class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state,runningStep)
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            
            state = nextState


        return trajectory

# class RollOut:
#     def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
#         self.transitionFunction = transitionFunction
#         self.rewardFunction = rewardFunction
#         self.maxRolloutStep = maxRolloutStep
#         self.rolloutPolicy = rolloutPolicy
#         self.isTerminal = isTerminal
#         self.rolloutHeuristic = rolloutHeuristic

#     def __call__(self, leafNode):
#         currentState = list(leafNode.id.values())[0]
#         totalRewardForRollout = 0

#         if leafNode.is_root:
#             lastState=currentState
#         else:
#             lastState=list(leafNode.parent.id.values())[0]

#         for rolloutStep in range(self.maxRolloutStep):
#             action = self.rolloutPolicy(currentState)
#             totalRewardForRollout += self.rewardFunction(lastState,currentState, action)
#             if self.isTerminal(lastState,currentState):
#                 break
#             nextState = self.transitionFunction(currentState, action)
#             lastState=currentState
#             currentState = nextState

#         heuristicReward = 0
#         if not self.isTerminal(lastState,currentState):
#             heuristicReward = self.rolloutHeuristic(currentState)
#         totalRewardForRollout += heuristicReward

#         return totalRewardForRollout

# class IsTerminal():
#     def __init__(self, getPredatorPos, getPreyPos, minDistance,divideDegree):
#         self.getPredatorPos = getPredatorPos
#         self.getPreyPos = getPreyPos
#         self.minDistance = minDistance
#         self.divideDegree=divideDegree
#     def __call__(self, lastState,currentState):
#         terminal = False

#         getPositionList=lambda getPos,lastState,currentState:np.linspace(getPos(lastState),getPos(currentState),self.divideDegree,endpoint=True)

#         getL2Normdistance= lambda preyPosition,predatorPosition :np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)

#         preyPositionList =getPositionList(self.getPreyPos,lastState,currentState)
#         predatorPositionList  = getPositionList(self.getPredatorPos,lastState,currentState)

#         L2NormdistanceList =[getL2Normdistance(preyPosition,predatorPosition) for (preyPosition,predatorPosition) in zip(preyPositionList,predatorPositionList) ]

#         if np.any(np.array(L2NormdistanceList) <= self.minDistance):
#             terminal = True
#         return terminal

# class Expand:
#     def __init__(self, isTerminal, initializeChildren):
#         self.isTerminal = isTerminal
#         self.initializeChildren = initializeChildren

#     def __call__(self, leafNode):
#         currentState = list(leafNode.id.values())[0]
#         if leafNode.is_root:
#             lastState=currentState
#         else:
#             lastState=list(leafNode.parent.id.values())[0]
#         if not self.isTerminal(lastState,currentState):
#             leafNode.isExpanded = True
#             leafNode = self.initializeChildren(leafNode)

#         return leafNode

# class SampleTrajectory:
#     def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
#         self.maxRunningSteps = maxRunningSteps
#         self.transit = transit
#         self.isTerminal = isTerminal
#         self.reset = reset
#         self.chooseAction = chooseAction

#     def __call__(self, policy):
#         state = self.reset()

#         while self.isTerminal(state,state):
#             state = self.reset()
#             print(L2NormdistanceList)
#             print(np.array(L2NormdistanceList) <= self.minDistance)

#         trajectory = []
#         lastState=state
#         for runningStep in range(self.maxRunningSteps):
#             if self.isTerminal(lastState,state):
#                 trajectory.append((state, None, None))
#                 break
#             actionDists = policy(state)
#             action = [self.chooseAction(actionDist) for actionDist in actionDists]
#             trajectory.append((state, action, actionDists))
#             nextState = self.transit(state, action)
#             lastState=state
#             state = nextState

#         return trajectory

# class SampleTrajectoryWithRender:
#     def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, render, renderOn):
#         self.maxRunningSteps = maxRunningSteps
#         self.transit = transit
#         self.isTerminal = isTerminal
#         self.reset = reset
#         self.chooseAction = chooseAction
#         self.render = render
#         self.renderOn = renderOn

#     def __call__(self, policy):
#         state = self.reset()

#         while self.isTerminal(state,state):
#             state = self.reset()

#         trajectory = []
#         lastState=state
#         for runningStep in range(self.maxRunningSteps):
#             if self.isTerminal(lastState,state):
#                 trajectory.append((state, None, None))
#                 break
#             if self.renderOn:
#                 self.render(state,runningStep)
#             actionDists = policy(state)
#             action = [self.chooseAction(actionDist) for actionDist in actionDists]
#             trajectory.append((state, action, actionDists))
#             nextState = self.transit(state, action)
#             lastState=state
#             state = nextState


#         return trajectory

# class RewardFunctionCompete():
#     def __init__(self, aliveBonus, deathPenalty, isTerminal):
#         self.aliveBonus = aliveBonus
#         self.deathPenalty = deathPenalty
#         self.isTerminal = isTerminal

#     def __call__(self, lastState,currentState, action):
#         reward = self.aliveBonus
#         if self.isTerminal(lastState,currentState):
#             reward += self.deathPenalty

#         return reward
if __name__ == '__main__':
    main()
