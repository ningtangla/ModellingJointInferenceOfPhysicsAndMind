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
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel

from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def main():
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsCenterControl', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 150
    numSimulations = 100
    killzoneRadius = 30
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    # parametersForTrajectoryPath = json.loads(sys.argv[1])
    # startSampleIndex = int(sys.argv[2])
    # endSampleIndex = int(sys.argv[3])
    # parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # numTrajectoriesPerIteration=parametersForTrajectoryPath['numTrajectoriesPerIteration']
    # numTrainStepEachIteration=parametersForTrajectoryPath['numTrainStepEachIteration']
    # selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    # otherIteration = int(parametersForTrajectoryPath['otherIteration'])
    parametersForTrajectoryPath = {}
    startSampleIndex = 0
    endSampleIndex = 8
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    parametersForTrajectoryPath['numTrajectoriesPerIteration']=16
    numTrajectoriesPerIteration=parametersForTrajectoryPath['numTrajectoriesPerIteration']

    parametersForTrajectoryPath['numTrainStepEachIteration']=1
    numTrainStepEachIteration=parametersForTrajectoryPath['numTrainStepEachIteration']
    parametersForTrajectoryPath['selfIteration']=40
    parametersForTrajectoryPath['otherIteration']=74

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    otherIteration = int(parametersForTrajectoryPath['otherIteration'])
    if not os.path.isfile(trajectorySavePath):


        numOfAgent=3
        sheepId = 0
        wolvesId = 1


        wolfOnePosIndex = 1
        wolfTwoIndex = 2

        agentIds = list(range(numOfAgent))
        xPosIndex = [0, 1]
        xBoundary = [0,600]
        yBoundary = [0,600]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOnePosIndex, xPosIndex)
        getWolfTwoXPos =GetAgentPosFromState(wolfTwoIndex, xPosIndex)

        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepAliveBonus = 1/maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]
        
        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
        isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)


        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary) 
        transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        

        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolvesActionSpace =list(product(wolfActionOneSpace,wolfActionTwoSpace))
        actionSpaceList=[sheepActionSpace,wolvesActionSpace]
        # neural network init
        numStateSpace = 6
        # numActionSpace = len(actionSpace)
        numSheepActionSpace=len(sheepActionSpace)
        numWolvesActionSpace=len(wolvesActionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        # generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolvesModel=GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        generateModelList=[generateSheepModel,generateWolvesModel]

        depth = 5
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        # multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in agentIds]
        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for generateModel in generateModelList]



        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsCenterControl', 'NNModelRes')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    
        startTime = time.time()
        trainableAgentIds = [sheepId, wolvesId]







        if otherIteration == -999:
            positionIndex = [0, 1]
            getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
            getPredatorPos = GetAgentPosFromState(wolvesId, positionIndex)
            wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)
        else:

            wolfModelPath = generateNNModelSavePath({'iterationIndex': otherIteration, 'agentId': wolvesId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
            restoredNNModel = restoreVariables(multiAgentNNmodel[wolvesId], wolfModelPath)
            multiAgentNNmodel[wolvesId] = restoredNNModel

            wolfPolicy = ApproximatePolicy(multiAgentNNmodel[wolvesId], wolvesActionSpace)

        # sample and save trajectories

        sheepModelPath = generateNNModelSavePath({'iterationIndex': selfIteration, 'agentId': sheepId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})

        sheepTrainedModel=restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)


        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)
        

        policy = lambda state:[sheepPolicy(state),wolfPolicy(state),]

        temperatureInMCTS = 1
        chooseActionInMCTS = SampleAction(temperatureInMCTS)
        chooseActionList = [chooseActionInMCTS, chooseActionInMCTS]
        # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

        from exec.evaluateNoPhysicsEnvWithRender import Render #SampleTrajectoryWithRender
        import pygame as pg 
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        saveImage = False
        saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        renderOn = True
        render=None
        if renderOn:
            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, xPosIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)



        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print(trajectories)
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
            actionFortransit=[action[0],action[1][0],action[1][1]]
            nextState = self.transit(state, actionFortransit)
            
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
