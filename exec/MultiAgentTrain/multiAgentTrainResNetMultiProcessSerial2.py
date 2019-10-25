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
import pathos.multiprocessing as mp
import itertools as it
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
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
    def __init__(self, numSimulations, actionSpace, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numSimulations = numSimulations
        self.actionSpace = actionSpace
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy(selfNNModel)
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace, transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue(selfNNModel)
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
        multiAgentApproximatePolicy = np.array([self.approximatePolicy(NNModel) for NNModel in multiAgentNNModel])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]])
                                                  for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
                                     for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy


class PreprocessTrajectoriesForBuffer:
    def __init__(self, addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory):
        self.addMultiAgentValuesToTrajectory = addMultiAgentValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addMultiAgentValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        return filteredTrajectories


class TrainOneAgent:
    def __init__(self, numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets,
                 sampleBatchFromBuffer, trainNN):
        self.numTrainStepEachIteration = numTrainStepEachIteration
        self.numTrajectoriesToStartTrain = numTrajectoriesToStartTrain
        self.sampleBatchFromBuffer = sampleBatchFromBuffer
        self.processTrajectoryForPolicyValueNets = processTrajectoryForPolicyValueNets
        self.trainNN = trainNN

    def __call__(self, agentId, multiAgentNNmodel, updatedReplayBuffer):
        NNModel = multiAgentNNmodel[agentId]
        if len(updatedReplayBuffer) >= self.numTrajectoriesToStartTrain:
            for _ in range(self.numTrainStepEachIteration):
                sampledBatch = self.sampleBatchFromBuffer(updatedReplayBuffer)
                processedBatch = self.processTrajectoryForPolicyValueNets[agentId](sampledBatch)
                trainData = [list(varBatch) for varBatch in zip(*processedBatch)]
                updatedNNModel = self.trainNN(NNModel, trainData)
                NNModel = updatedNNModel

        return NNModel
def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [1,4,16]
    manipulatedVariables['numTrajectoriesPerIteration'] = [1,4,16]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = int(0.8*numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    #trainedModels = [trainPool.apply_async(trainModelForConditions, (parameters,)) for parameters in parametersAllCondtion]
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)

def iterateTrainOneCondition(parameters):
    # Mujoco environment
    numTrainStepEachIteration = int(parameters['numTrainStepEachIteration'])
    numTrajectoriesPerIteration = int(parameters['numTrajectoriesPerIteration'])
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    # MDP function
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 8
    qPosInitNoise = 9.7
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    agentIds = list(range(numAgents))
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    maxRunningSteps = 25
    sheepAliveBonus = 0.05
    wolfAlivePenalty = -sheepAliveBonus

    sheepTerminalPenalty = -1
    wolfTerminalReward = 1
    terminalRewardList = [sheepTerminalPenalty, wolfTerminalReward]

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

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
    temperatureInMCTS = 1
    chooseActionInMCTS = SampleAction(temperatureInMCTS)
    chooseActionList = [chooseActionInMCTS, chooseActionInMCTS]
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionList)

    # neural network init
    numStateSpace = 12
    numActionSpace = len(actionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [256]
    actionLayerWidths = [256]
    valueLayerWidths = [256]
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

    trainReporter = TrainReporter(numTrainStepEachIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRate = 0.0001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    trainNN = Train(numTrainStepEachIteration, miniBatchSize, sampleData,
                    learningRateModifier, terminalController, coefficientController,
                    trainReporter)

    # load save dir
    numSimulations = 100
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'multiAgentTrain', 'multiMCTSAgentResNet', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgentResNet', 'NNModelRes')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    #frequencyVersion: delete used model for disk space
    toDeleteNNModelExtensionList=['.meta','.index','.data-00000-of-00001']
    generatetoDeleteNNModelPathList=[GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParameters) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]



    startTime = time.time()
    trainableAgentIds = [sheepId, wolfId]

    depth = 17
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for agentId in agentIds]

    temperatureInMCTS = 1
    chooseActionInMCTS = SampleAction(temperatureInMCTS)

    otherAgentApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)

    composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS)
    composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numSimulations, actionSpace, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS)
    prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)
    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory)
    numTrajectoriesToStartTrain = 4 * miniBatchSize

    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)

    restoredIteration = 0

    for agentId in trainableAgentIds:
        modelPathBeforeTrain = generateNNModelSavePath({'iterationIndex': 0, 'agentId': agentId})
        saveVariables(multiAgentNNmodel[agentId], modelPathBeforeTrain)


    sampleTrajectoryFileName = 'sampleMultiMCTSAgentResNetTrajCondtion.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8*numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    trajectoryBeforeTrainPathParamters = {'iterationIndex': 0}
    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForParallel = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    # load trajectory function for trainBreak
    loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)

# initRreplayBuffer
    replayBuffer = []


    # if restoredIteration == 0:
    #     cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

# restore model
    for agentId in trainableAgentIds:
        if restoredIteration == 0:
            modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId})
        else:
            modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId,  'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters={}, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)


    numIterations = 1000
    modelSaveFrequency=25

    for iterationIndex in range(restoredIteration + 1, numIterations):
        print("ITERATION INDEX: ", iterationIndex)

        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        trajectories = [sampleTrajectory(policy) for _ in range(numTrajectoriesPerIteration)]

        numCpuToUseWhileTrain = int(2)
        numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
        # generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)

        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}

        # cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)
        # trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
        # print('length of traj', len(trajectories))
        # trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        # saveToPickle(trajectories, trajectorySavePath)

        # add trajctroy check
        # trajecoriesNum=0
        # while trajecoriesNum!=numTrajectoriesPerIteration:
        #     cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)

        #     trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
        #     trajecoriesNum=len(trajectories)
        #     if trajecoriesNum!=numTrajectoriesPerIteration:
        #         print('MISSSUBPROCESS,RETRY',trajecoriesNum)
        # print('length of traj', len(trajectories))
        trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        saveToPickle(trajectories, trajectorySavePath)

        preProcessedTrajectories = preprocessMultiAgentTrajectories(trajectories)
        updatedReplayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectories)

        for agentId in trainableAgentIds:

            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)

            #tempversion: save tempModel for generate traj
            # tempNNModelPathParameters = {'temp': 1, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
            # tempNNModelSavePath = generateNNModelSavePath(tempNNModelPathParameters)
            # saveVariables(updatedAgentNNModel, tempNNModelSavePath)
            # if iterationIndex % modelSaveFrequency == 0:
            #     NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
            #     NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            #     saveVariables(updatedAgentNNModel, NNModelSavePath)


            #frequencyVersion: delete used model for disk space
            if iterationIndex % modelSaveFrequency != 0 and iterationIndex>=modelSaveFrequency:
                toDeleteNNModelPathParameters={'iterationIndex': iterationIndex-modelSaveFrequency, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
                toDeleteModelPathList = [generatetoDeleteNNModelPath(toDeleteNNModelPathParameters) for generatetoDeleteNNModelPath in generatetoDeleteNNModelPathList ]
                for toDeleteModelPath in toDeleteModelPathList:
                    # print('todelete:',toDeleteModelPath)
                    if os.path.isfile(toDeleteModelPath):
                        os.remove(toDeleteModelPath)
                        print('delete',toDeleteModelPath)
                    else:
                        print('no such file',toDeleteModelPath)
            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration':numTrajectoriesPerIteration, 'numTrainStepEachIteration':numTrainStepEachIteration}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)


            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()
