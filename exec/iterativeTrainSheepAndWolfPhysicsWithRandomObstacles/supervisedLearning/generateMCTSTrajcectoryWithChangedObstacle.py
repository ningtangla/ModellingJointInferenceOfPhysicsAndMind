import time
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..','..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco
import xmltodict

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.envMujocoRandomObstacles import SampleObscalesProperty,  SetMujocoEnvXmlProperty, changeWallProperty,TransitionFunction,CheckAngentStackInWall,ResetUniformInEnvWithObstacles,getWallList
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,HeuristicDistanceToTarget

from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup,Expand,RollOut,establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy,RandomPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction

from exec.parallelComputing import GenerateTrajectoriesParallel

def main():

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])

    # numSimulations = int(parametersForTrajectoryPath['numSimulations'])
    numSimulations=int(parametersForTrajectoryPath['numSimulations'])
    maxRolloutSteps=int(parametersForTrajectoryPath['maxRolloutSteps'])
    agentId=int(parametersForTrajectoryPath['agentId'])
    # check file exists or not
    dirName = os.path.dirname(__file__)

    dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'MCTSFixObstacle')
    trajectoriesSaveDirectory = os.path.join(dataFolderName,  'trajectories')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    killzoneRadius = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius,'agentId':agentId}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        physicsDynamicsPath=os.path.join(dirName,'..','..','..','env','xmls','twoAgentsTwoRandomObstacles.xml')
        with open(physicsDynamicsPath) as f:
            xml_string = f.read()
        originalEnvXmlDict = xmltodict.parse(xml_string.strip())

        wallIDlist=[5,6]
        wall1Pos=[0,2.5,-0.2]
        wall1Size=[0.8,1.95,1.5]
        wall2Pos=[0,-2.5,-0.2]
        wall2Size=[0.8,1.95,1.5]
        initWallPosList=[wall1Pos,wall2Pos]
        initWallSizeList=[wall1Size,wall2Size]

        wallXdelta=[-8,8]
        wallYdelta=[-3.8,4.9]
        class RandomMoveObstacleCenter(object):
            def __init__(self, initWallPosList,wallXdelta,wallYdelta):
                self.initWallPosList = initWallPosList
                self.wallXdelta = wallXdelta
                self.wallYdelta = wallYdelta
            def __call__(self):
                x=np.random.uniform(self.wallXdelta,size=1)[0]
                y=np.random.uniform(self.wallYdelta,size=1)[0]
                movingVector=[x,y,0]
                wallPosList=[[pos+delta for pos,delta in zip(obstacle,movingVector)] for obstacle in self.initWallPosList]
                print(movingVector,wallPosList)
                return wallPosList
        # def randomMoveObstacleCenter(initWallPosList,wallXdelta,wallYdelta)
        #     x=np.random.uniform(wallXdelta,size=1)[0]
        #     y=np.random.uniform(wallYdelta,size=1)[0]
        #     movingVector=[x,y,0]
        #     wallPosList=[[pos+delta for pos,delta in zip(obstacle,movingVector)] for obstacle in initWallPosList]
        #     print(movingVector,wallPosList)
        #     return wallPosList
        sampleMovedWallPos=RandomMoveObstacleCenter(initWallPosList,wallXdelta,wallYdelta)
        # sampleFixWallPos=lambda:initWallPosList
        sampleFixWallSize=lambda:wallSizeList


        sampleObscalesProperty=SampleObscalesProperty(sampleMovedWallPos,sampleFixWallSize)
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

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        sheepAliveBonus = 1 / maxRunningSteps
        wolfAlivePenalty = -sheepAliveBonus
        sheepTerminalPenalty = -1
        wolfTerminalReward = 1
        terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        rewardWolf = RewardFunctionCompete(wolfAlivePenalty, wolfTerminalReward, isTerminal)
        rewardMultiAgents = [rewardSheep, rewardWolf]

        numSimulationFrames = 20
        transit = TransitionFunction(numAgents,physicsSimulation , numSimulationFrames,isTerminal)


        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

        randomSheepPolicy=RandomPolicy(actionSpace)
        sheepPolicy=randomSheepPolicy


        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

        def wolvesTransit(state, action):
             return transit(
            state, [chooseGreedyAction(sheepPolicy(state)),action])

        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            actionSpace, wolvesTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        numWolfActionSpace=len(actionSpace)
        def rolloutPolicy(
            state): return actionSpace[np.random.choice(range(numWolfActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0.0
        rolloutHeuristic = HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfXPos, getSheepXPos)



        rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit,rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # randomWolfPolicy=RandomPolicy(actionSpace)
        # wolfPolicy=randomWolfPolicy

        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]

        # sample trajectory
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
