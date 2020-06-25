import time
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco
import xmltodict

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.envMujocoRandomObstacles import SampleObscalesProperty,  SetMujocoEnvXmlProperty, changeWallProperty,TransitionFunction,CheckAngentStackInWall,ResetUniformInEnvWithObstacles,getWallList
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from exec.iterativeTrainSheepAndWolfPhysicsWithRandomObstacles.NNGuidedMCTS import ApproximatePolicy,ApproximateValue

from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, SampleAction, chooseGreedyAction

from exec.parallelComputing import GenerateTrajectoriesParallel

def main():

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    depth=int(parametersForTrajectoryPath['depth'])
    learningRate=float(parametersForTrajectoryPath['learningRate'])
    selfIteration = int(parametersForTrajectoryPath['selfIteration'])
    otherIteration = int(parametersForTrajectoryPath['otherIteration'])


    # check file exists or not
    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentFixObstacle')
    trajectoriesSaveDirectory = os.path.join(dataFolderName,  'evaluateTrajectories')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 30
    numSimulations = 200
    killzoneRadius=2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        physicsDynamicsPath=os.path.join(dirName,'..','..','env','xmls','twoAgentsTwoRandomObstacles.xml')
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
        setMujocoEnvXmlProperty=SetMujocoEnvXmlProperty(wallIDlist,changeWallProperty)

        wallPosList,wallSizeList=sampleObscalesProperty()
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
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # neural network init
        numStateSpace = 20
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        initWolfNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)
        initSheepNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)

        # load save dir
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dataFolderName, 'NNModel')
        getNNModelSavePath =  GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)
        wolfModelPath = getNNModelSavePath({'iterationIndex': selfIteration, 'agentId': wolfId, 'depth':depth, 'learningRate':learningRate})
        sheepModelPath = getNNModelSavePath({'iterationIndex': selfIteration, 'agentId': sheepId, 'depth':depth, 'learningRate':learningRate})

        restoreWolfNNModel = restoreVariables(initWolfNNModel, wolfModelPath)
        restoreSheepNNModel=restoreVariables(initSheepNNModel, sheepModelPath)

        sheepPolicy = ApproximatePolicy(restoreSheepNNModel, actionSpace)
        wolfPolicy = ApproximatePolicy(restoreWolfNNModel, actionSpace)


        # sample and save trajectories
        policy = lambda state:[sheepPolicy(state),wolfPolicy(state)]
        # sample trajectory
        chooseActionList = [chooseGreedyAction, chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]

        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()