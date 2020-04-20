import time
import sys
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))


import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories,  GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer

from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from exec.parallelComputing import GenerateTrajectoriesParallel

def iterateTrainOneCondition(parameters):
    

    numTrainStepEachIteration = 1
    numTrajectoriesPerIteration = 4
    learningRate=parameters['learningRate']
    dept1=parameters['depth']
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = physicsDynamicsPath=os.path.join(dirName,'twoAgentsTwoObstacles3.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    agentIds = list(range(numAgents))
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)


    maxRunningSteps = 30

    # NNGuidedMCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]


    # neural network init
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # replay buffer
    bufferSize = 2000
    saveToBuffer = SaveToBuffer(bufferSize)
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 256
    sampleBatchFromBuffer = SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # pre-process the trajectory for replayBuffer
    addMultiAgentValuesToTrajectory = AddValuesToTrajectory(accumulateMultiAgentRewards)
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    # pre-process the trajectory for NNTraining
    actionToOneHot = ActionToOneHot(actionSpace)
    processTrajectoryForPolicyValueNets = [ProcessTrajectoryForPolicyValueNetMultiAgentReward(actionToOneHot, agentId) for agentId in agentIds]

    # function to train NN model
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
    reportInterval = 1
    numTrainStepEachIteration = 1
    
    trainReporter = TrainReporter(numTrainStepEachIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    # learningRate = 0.001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    trainNN = Train(numTrainStepEachIteration, miniBatchSize, sampleData,
                    learningRateModifier, terminalController, coefficientController,
                    trainReporter)

    # load save dir
    dataFolderName=os.path.join('..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentFixObstacle')

    numSimulations = 200
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    
    trajectoriesSaveDirectory = os.path.join(dataFolderName, 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dataFolderName, 'NNModel')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    #frequencyVersion: delete used model for disk space
    toDeleteNNModelExtensionList=['.meta','.index','.data-00000-of-00001']
    generatetoDeleteNNModelPathList=[GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParameters) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]

    startTime = time.time()
    trainableAgentIds = [sheepId, wolfId]


    startTime = time.time()
    trainableAgentIds = [sheepId, wolfId]

    depth = 4
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths) for agentId in agentIds]
    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory)
    numTrajectoriesToStartTrain = 4 * miniBatchSize

    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)


    # save step 0 Model for evaluate
    for agentId in trainableAgentIds:
        NNModelPathParameters = {'iterationIndex': 0, 'agentId': agentId, 'depth':depth, 'learningRate':learningRate}
        NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
        saveVariables(multiAgentNNmodel[agentId], NNModelSavePath)
    
    # initRreplayBuffer
    replayBuffer = []
    trajectoryBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': trajectoryBeforeTrainIndex}
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

    # restore modelrestoredIteration
    restoredIteration=0#0
    for agentId in trainableAgentIds:
        modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId,  'depth':depth, 'learningRate':learningRate})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)


    #paralle sample trajectory
    numCpuToUseWhileTrain = int(16)
    numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
    sampleTrajectoryFileName = 'sampleMultiMCTSAgentTrajectoryFixedObstacle.py'
    generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)

    # delete used model for disk space
    fixedParametersForDelete = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius, 'depth':depth, 'learningRate':learningRate}
    toDeleteNNModelExtensionList = ['.meta', '.index', '.data-00000-of-00001']
    generatetoDeleteNNModelPathList = [GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParametersForDelete) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]

    modelMemorySize = 5
    modelSaveFrequency = 1000
    deleteUsedModel = DeleteUsedModel(modelMemorySize, modelSaveFrequency, generatetoDeleteNNModelPathList)
    numIterations = 1000
    numTrajectoriesPerIteration = 4

    for iterationIndex in range(restoredIteration + 1, numIterations):

        trajectoryPathParameters = {'iterationIndex': iterationIndex, depth':depth, 'learningRate':learningRate'}

        trajecoriesNum=0
        while trajecoriesNum!=numTrajectoriesPerIteration:
            cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)

            trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
            trajecoriesNum=len(trajectories)
            if trajecoriesNum!=numTrajectoriesPerIteration:
                print('MISSSUBPROCESS,RETRY',trajecoriesNum)
        print('length of traj', len(trajectories))
        trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        saveToPickle(trajectories, trajectorySavePath)

        preProcessedTrajectories = preprocessMultiAgentTrajectories(trajectories)
        updatedReplayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectories)

        for agentId in trainableAgentIds:
            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)

            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'depth':depth, 'learningRate':learningRate}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)
            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

            deleteUsedModel(iterationIndex, agentId)
    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [4]
    manipulatedVariables['numTrajectoriesPerIteration'] = [16]
    manipulatedVariables['depth']=[4]
    manipulatedVariables['learningRate']=[ 0.001]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    #Sample Trajectory Before Train to fill Buffer
    miniBatchSize = 256
    numTrajectoriesToStartTrain = 4 * miniBatchSize
    sampleTrajectoryFileName = 'prepareMultiMCTSAgentTrajectoryFixObstacle.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8 * numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    iterationBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': iterationBeforeTrainIndex}
    prepareBefortrainData = True
    if prepareBefortrainData:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)

    #parallel train
    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)    