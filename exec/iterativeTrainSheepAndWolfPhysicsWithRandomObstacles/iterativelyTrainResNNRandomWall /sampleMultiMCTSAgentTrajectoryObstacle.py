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
import mujoco_py as mujoco
import xmltodict
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.envMujocoRandomObstacles import SampleObscalesProperty, SetMujocoEnvXmlProperty, changeWallProperty,TransitionFunction,CheckAngentStackInWall,ResetUniformInEnvWithObstacles,getWallList,RandomMoveObstacleCenter,SampleRandomWallSize,RejectSampleObstacleMoveVector,CheckObstacleOutEnv
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, \
    LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData,  restoreVariables

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, \
    LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode

from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.NNGuidedMCTS import ComposeMultiAgentTransitInSingleAgentMCTS,ComposeSingleAgentGuidedMCTS,PrepareMultiAgentPolicy,ApproximatePolicy,ApproximateValue

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel


def main():
    # check file exists or not

    dirName = os.path.dirname(__file__)
    dataDirectory = os.path.join(dirName, '..', '..', '..', 'data','multiAgentTrain', 'multiMCTSAgentResNNRandomObstacle')
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    trajectoriesSaveDirectory = os.path.join(dataDirectory, 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    numSimulations = 200
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,
                       'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath=os.path.join(dirName,'..','..','..','env','xmls','twoAgentsTwoRandomObstacles.xml')
        with open(physicsDynamicsPath) as f:
           xml_string = f.read()
        originalEnvXmlDict = xmltodict.parse(xml_string.strip())

        wallIDlist=[5,6]

        gapDelta=[0.55,1.2]
        wallLengthDelta=[0.8,2.5]
        wallWidthDelta=[0.8,2.5]
        sampleRandomWallSize=SampleRandomWallSize(gapDelta,wallLengthDelta,wallWidthDelta)

        initWallPosList,initWallSizeList=sampleRandomWallSize()
        
        wallXdelta=[-10,10]
        wallYdelta=[-10,10]
        sampleMovedWallPos=RandomMoveObstacleCenter(initWallPosList,wallXdelta,wallYdelta)
        
        allowedArea=[[-8,8],[-8,8]]
        checkObstacleInEnv=CheckObstacleOutEnv(initWallSizeList,allowedArea)
        rejectSampleObstacleMoveVector=RejectSampleObstacleMoveVector(sampleMovedWallPos,checkObstacleInEnv)

        sampleFixWallSize=lambda:initWallSizeList
        sampleObscalesProperty=SampleObscalesProperty(rejectSampleObstacleMoveVector,sampleFixWallSize)

        wallPosList,wallSizeList=sampleObscalesProperty()

        setMujocoEnvXmlProperty=SetMujocoEnvXmlProperty(wallIDlist,changeWallProperty)
        envXmlDict=setMujocoEnvXmlProperty(wallPosList,wallSizeList,originalEnvXmlDict)

        envXml=xmltodict.unparse(envXmlDict)
        physicsModel = mujoco.load_model_from_xml(envXml)
        physicsSimulation = mujoco.MjSim(physicsModel)

        agentMaxSize=0.6
        wallList=getWallList(wallPosList,wallSizeList)
        checkAngentStackInWall=CheckAngentStackInWall(agentMaxSize)

       # MDP function
        qPosInit = (0, 0, 0, 0)
        qVelInit = [0, 0, 0, 0]
        numAgents = 2
        qVelInitNoise = 8
        qPosInitNoise = 9.7

        reset=ResetUniformInEnvWithObstacles(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise,wallList,checkAngentStackInWall)

        agentIds = list(range(numAgents))
        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus

        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(numAgents,physicsSimulation , numSimulationFrames,isTerminal)

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        decay = 1
        accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay, rewardMultiAgents)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        getApproximatePolicy = lambda NNmodel: ApproximatePolicy(NNmodel, actionSpace)
        getApproximateValue = lambda NNmodel: ApproximateValue(NNmodel)

        getStateFromNode = lambda node: list(node.id.values())[0]

        # sample trajectory
        chooseActionList=[chooseGreedyAction,chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

        # neural network init
        numStateSpace = 20
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        # NNModelSaveDirectory = os.path.join(dataDirectory, 'NNModel')


        NNModelSaveDirectory = os.path.join(dirName, dataDirectory, 'ResNNModel')

        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)


        trainableAgentIds = [sheepId, wolfId]

        depth = 9
        resBlock = 2
        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths,resBlock) for agentId in agentIds]


        startTime = time.time()



        otherAgentApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseGreedyAction)
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpace, terminalRewardList,selectChild, isTerminal, transit, getStateFromNode,getApproximatePolicy, getApproximateValue,composeMultiAgentTransitInSingleAgentMCTS)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy,trainableAgentIds)


        # load NN
        # multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for agentId in agentIds]
        

        for agentId in trainableAgentIds:
            modelPath = generateNNModelSavePath({'iterationIndex': iterationIndex, 'agentId': agentId})
            restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPath)
            multiAgentNNmodel[agentId] = restoredNNModel

        # sample and save trajectories
        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
