import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product 
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysicsWithCenterControlAction, Reset,IsTerminal,StayInBoundaryByReflectVelocity,UnpackCenterControlAction
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
from src.episode import Render,SampleTrajectoryWithRender, SampleAction, chooseGreedyAction
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
 
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    isImagedSampleActionGreedy=int(parametersForTrajectoryPath['isImagedSampleActionGreedy'])
    isTansitSampleActionGreedy=int(parametersForTrajectoryPath['isTansitSampleActionGreedy'])
    
    np.random.seed(startSampleIndex*endSampleIndex)
    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..','..', 'data','multiAgentTrain', 'evaluateCeterControlNNGuidedMCTSSampleAction', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    #get traj save path
    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 100
    numSimulations = 200
    killzoneRadius = 30
    # fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'isPureMCTS':0}
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        #No physics env
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        posIndex = [0, 1]
        getSheepXPos = GetAgentPosFromState(sheepId, posIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, posIndex)
        getWolfTwoXPos =GetAgentPosFromState(wolfTwoId, posIndex)

        numOfAgent=3
        xBoundary = [0,600]
        yBoundary = [0,600]
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        sheepAliveBonus = 1/maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
        isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)
        
        wolvesId =1
        centerControlIndexList=[wolvesId]
        unpackAction=UnpackCenterControlAction(centerControlIndexList)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary) 
        transit=TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity,unpackAction)

        #product wolves action space 
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7),(0,0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolvesActionSpace =list(product(wolfActionOneSpace,wolfActionTwoSpace))
        actionSpaceList=[sheepActionSpace,wolvesActionSpace]

        # neural network init
        numStateSpace = 6
        numSheepActionSpace=len(sheepActionSpace)
        numWolvesActionSpace=len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        generateWolvesModel=GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        generateModelList=[generateSheepModel,generateWolvesModel]
        
        sheepDepth = 5
        wolfDepth=9
        depthList=[sheepDepth,wolfDepth]
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        
        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList,generateModelList)]
        
        #restorePretrainModel
        sheepPreTrainModelPath=os.path.join(dirName,'..','preTrainModel','agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        wolvesPreTrainModelPath=os.path.join(dirName,'..','preTrainModel','agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        pretrainModelPathList=[sheepPreTrainModelPath,wolvesPreTrainModelPath]
        
        restoreAgentIds = [sheepId,wolvesId]
        for agentId in restoreAgentIds:

            restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], pretrainModelPathList[agentId])
            multiAgentNNmodel[agentId] = restoredNNModel

        otherAgentApproximatePolicy = [lambda NNmodel,: ApproximatePolicy(NNmodel, sheepActionSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolvesActionSpace)]

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getApproximatePolicy =[lambda NNmodel,: ApproximatePolicy(NNmodel, sheepActionSpace),lambda NNmodel,: ApproximatePolicy(NNmodel, wolvesActionSpace)]
        getApproximateValue = [lambda NNmodel: ApproximateValue(NNmodel),lambda NNmodel: ApproximateValue(NNmodel)]
        getStateFromNode = lambda node: list(node.id.values())[0]

        temperature = 1
        sampleAction = SampleAction(temperature)
        actionSampleList=[sampleAction,chooseGreedyAction]
        

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(actionSampleList[isImagedSampleActionGreedy])

        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS)

        trainableAgentIds=[wolvesId]
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)
        
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        # sheepPolicy = ApproximatePolicy(multiAgentNNmodel[sheepId], sheepActionSpace)
        
        # wolfPolicy=ApproximatePolicy(multiAgentNNmodel[wolvesId], wolvesActionSpace)

        # policy=lambda state:[sheepPolicy(state),wolfPolicy(state)]
        # sample and save trajectories
        chooseActionList = [chooseGreedyAction,actionSampleList[isTansitSampleActionGreedy]]

        saveImage = False
        saveImageDir = os.path.join(dirName, '..','..', '..', 'data','demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        render=None
        renderOn = False
        if renderOn:
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
            circleSize = 10
            screen = pg.display.set_mode([max(xBoundary), max(yBoundary)])
            render = Render(numOfAgent, posIndex,screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList,render,renderOn)

        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)



if __name__ == '__main__':
    main()
