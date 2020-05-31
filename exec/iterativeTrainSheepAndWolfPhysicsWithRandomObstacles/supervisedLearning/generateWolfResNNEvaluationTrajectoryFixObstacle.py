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

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup,Expand,RollOut,establishSoftmaxActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy,RandomPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction

from exec.parallelComputing import GenerateTrajectoriesParallel


from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue,ApproximatePolicy, restoreVariables

from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.NNGuidedMCTS import ApproximatePolicy,ApproximateValue

def main():

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    # iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])

    # numSimulations = int(parametersForTrajectoryPath['numSimulations'])
    depth=int(parametersForTrajectoryPath['depth'])
    miniBatchSize=int(parametersForTrajectoryPath['miniBatchSize'])
    learningRate=float(parametersForTrajectoryPath['learningRate'])
    trainSteps=int(parametersForTrajectoryPath['trainSteps'])
    # check file exists or not
    dirName = os.path.dirname(__file__)

    dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'MCTSFixObstacle')
    trajectoryDirectory = os.path.join(dataFolderName,'evaluationTrajectories')


    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    killzoneRadius = 2
    numSimulations=150
    agentId=1
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius,'agentId':agentId}

    generateTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectorySaveExtension, fixedParameters)
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
        wallPosList=[wall1Pos,wall2Pos]
        wallSizeList=[wall1Size,wall2Size]

        sampleFixWallPos=lambda:wallPosList
        sampleFixWallSize=lambda:wallSizeList

        sampleObscalesProperty=SampleObscalesProperty(sampleFixWallPos,sampleFixWallSize)
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



        numSimulationFrames = 20
        transit = TransitionFunction(numAgents,physicsSimulation , numSimulationFrames,isTerminal)


        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numactionSpace=len(actionSpace)
        randomSheepPolicy=RandomPolicy(actionSpace)
        sheepPolicy=randomSheepPolicy

        # neural network init and save path
        numStateSpace = 20
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        resBlockSize = 2
        dropoutRate = 0.0

        initializationMethod = 'uniform'
        generateModel = GenerateModel(numStateSpace, numactionSpace, regularizationFactor)
        initWolfNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        NNModelSaveDirectory = os.path.join(dataFolderName, 'trainedWolfResModels')

        NNModelSaveExtension=' '
        dataSetMaxRunningSteps=30
        dataSetNumSimulations=150
        NNModelFixedParameters = {'agentId': wolfId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

        parameters={'depth':depth,'learningRate':learningRate,'miniBatchSize':miniBatchSize,'trainSteps':trainSteps}
        wolfModelPath = getNNModelSavePath(parameters)

        restoreWolfNNModel = restoreVariables(initWolfNNModel, wolfModelPath)

        wolfPolicy = ApproximatePolicy(restoreWolfNNModel, actionSpace)
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]

        # sample trajectory
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
