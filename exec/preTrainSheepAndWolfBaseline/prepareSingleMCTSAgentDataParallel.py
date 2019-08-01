import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..'))
# import ipdb

import numpy as np
from collections import OrderedDict, deque
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import  SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel



def main():
    dirName = os.path.dirname(__file__)
    # load save dir
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'preTrainBaseline', 'trajectories')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    sheepId = 0
    wolfId = 1

    startTime = time.time()

    numTrajectories = 5000
    # generate and load trajectories before train parallelly
    sampleTrajectoryFileName = 'sampleMCTSWolfTrajectory.py'
    # sampleTrajectoryFileName = 'sampleMCTSSheepTrajectory.py'
    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.75*numCpuCores)
    numCmdList = min(numTrajectories, numCpuToUse)

    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectories, numCmdList)

    killzoneRadius = 2
    maxRunningSteps = 20
    numSimulations = 200
    fixedParameters = {'agentId':wolfId,'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
   
    print("start")
    trainableAgentIds = [sheepId]
    for agentId in trainableAgentIds:
        print("agent {}".format(agentId))
        pathParameters = {'agentId': agentId}

        cmdList = generateTrajectoriesParallel(pathParameters)
        # print(cmdList)
        # import ipdb; ipdb.set_trace()
    
    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)    
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)


    trajectories = loadTrajectories(parameters={})
    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))
    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]

    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    depth = 4
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    NNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)

    trainDataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    # function to train NN model
    batchSize = 256
    learningRate = 0.001
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 1000#1000
    trainStepsIntervel = 100000 #100000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier, terminalController, coefficientController,trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'preTrainBaseline',
                                        'trainedModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)
    trainingParameters = {'learningRate':learningRate, 'miniBatchSize': batchSize, 'depth':depth, 'trainSteps':trainStepsIntervel}

    trainedNN = getTrainNN(NNModel, trainData)
    saveVariables(trainedNN, getNNModelSavePath(trainingParameters))

if __name__ == '__main__':
    main()
