import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValueFunction, \
    ApproximateActionPrior, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, NNModel):
        wolfPolicy = self.getWolfPolicy(NNModel)
        sheepPolicy = self.getSheepPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class QPosInitStdDevForIteration:
    def __init__(self, minIterLinearRegion, stdDevMin, maxIterLinearRegion, stdDevMax):
        self.minIterLinearRegion = minIterLinearRegion
        self.stdDevMin = stdDevMin
        self.maxIterLinearRegion = maxIterLinearRegion
        self.stdDevMax = stdDevMax

    def __call__(self, iteration):
        if iteration < self.minIterLinearRegion:
            stdDev = self.stdDevMin
        elif iteration > self.maxIterLinearRegion:
            stdDev = self.stdDevMax
        else:
            constant = (self.stdDevMin * self.maxIterLinearRegion - self.stdDevMax * self.minIterLinearRegion) / (self.maxIterLinearRegion - self.minIterLinearRegion)
            stdDev = constant + (self.stdDevMax - self.stdDevMin) / (self.maxIterLinearRegion - self.minIterLinearRegion) * iteration

        return (0, 0, stdDev, stdDev)


class GetSampleTrajectoryForIteration:
    def __init__(self, getQPosInitStdDev, getReset, getSampleTrajectoryFromReset):
        self.getQPosInitStdDev = getQPosInitStdDev
        self.getReset = getReset
        self.getSampleTrajectoryFromReset = getSampleTrajectoryFromReset

    def __call__(self, iteration):
        qPosInitStdDev = self.getQPosInitStdDev(iteration)
        reset = self.getReset(qPosInitStdDev)
        sampleTrajectory = self.getSampleTrajectoryFromReset(reset)

        return sampleTrajectory


class GenerateTrajectories:
    def __init__(self, numTrajectoriesPerIteration, sampleTrajectory, preparePolicy, saveAllTrajectories):
        self.numTrajectoriesPerIteration = numTrajectoriesPerIteration
        # self.getSampleTrajectoryForIteration = getSampleTrajectoryForIteration
        self.sampleTrajectory = sampleTrajectory
        self.preparePolicy = preparePolicy
        self.saveAllTrajectories = saveAllTrajectories

    def __call__(self, iteration, NNModel, pathParameters):
        policy = self.preparePolicy(NNModel)
        # sampleTrajectory = self.getSampleTrajectoryForIteration(iteration)
        # trajectories = [sampleTrajectory(policy) for trial in range(self.numTrajectoriesPerIteration)]
        trajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrajectoriesPerIteration)]
        self.saveAllTrajectories(trajectories, pathParameters)

        return None


class ProcessTrajectoryForNN:
    def __init__(self, actionToOneHot, agentId):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*triple) for triple in trajectory]

        return processedTrajectory


class PreProcessTrajectories:
    def __init__(self, addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN):
        self.addValuesToTrajectory = addValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory
        self.processTrajectoryForNN = processTrajectoryForNN

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        processedTrajectories = [self.processTrajectoryForNN(trajectory) for trajectory in filteredTrajectories]

        return processedTrajectories


class IterativePlayAndTrain:
    def __init__(self, numIterations, learningThresholdFactor, saveNNModel, getGenerateTrajectories,
                 preProcessTrajectories, getSampleBatchFromBuffer, getTrainNN, getNNModel, loadTrajectories,
                 getSaveToBuffer, generatePathParametersAtIteration, getModelSavePath, restoreVariables):
        self.numIterations = numIterations
        self.learningThresholdFactor = learningThresholdFactor
        self.saveNNModel = saveNNModel
        self.getGenerateTrajectories = getGenerateTrajectories
        self.preProcessTrajectories = preProcessTrajectories
        self.getSampleBatchFromBuffer = getSampleBatchFromBuffer
        self.getTrainNN = getTrainNN
        self.getNNModel = getNNModel
        self.loadTrajectories = loadTrajectories
        self.getSaveToBuffer = getSaveToBuffer
        self.generatePathParametersAtIteration = generatePathParametersAtIteration
        self.getModelSavePath = getModelSavePath
        self.restoreVariables = restoreVariables

    def __call__(self, oneConditionDf):
        numTrajectoriesPerIteration = oneConditionDf.index.get_level_values('numTrajectoriesPerIteration')[0]
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]
        bufferSize = oneConditionDf.index.get_level_values('bufferSize')[0]
        generateTrajectories = self.getGenerateTrajectories(numTrajectoriesPerIteration)
        sampleBatchFromBuffer = self.getSampleBatchFromBuffer(miniBatchSize)
        trainNN = self.getTrainNN(learningRate)
        NNModel = self.getNNModel()
        buffer = []
        saveToBuffer = self.getSaveToBuffer(bufferSize)
        startTime = time.time()

        for iterationIndex in range(self.numIterations):
            print("ITERATION INDEX: ", iterationIndex)
            pathParametersAtIteration = self.generatePathParametersAtIteration(oneConditionDf, iterationIndex)
            modelSavePath = self.getModelSavePath(pathParametersAtIteration)
            # if not os.path.isfile(modelSavePath+'.index'):
            self.saveNNModel(NNModel, pathParametersAtIteration)
            generateTrajectories(iterationIndex, NNModel, pathParametersAtIteration)
            trajectories = self.loadTrajectories(pathParametersAtIteration)
            processedTrajectories = self.preProcessTrajectories(trajectories)
            updatedBuffer = saveToBuffer(buffer, processedTrajectories)
            if len(updatedBuffer) >= self.learningThresholdFactor * miniBatchSize:
                # if os.path.isfile(modelSavePath + '.index'):
                #     updatedNNModel = self.restoreVariables(NNModel, modelSavePath)
                #     NNModel = updatedNNModel
                # else:
                sampledBatch = sampleBatchFromBuffer(updatedBuffer)
                trainData = [list(varBatch) for varBatch in zip(*sampledBatch)]
                updatedNNModel = trainNN(NNModel, trainData)
                NNModel = updatedNNModel
            buffer = updatedBuffer

        endTime = time.time()
        print("Time taken for {} iterations: {} seconds".format(self.numIterations, (endTime - startTime)))


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(
            self.actionSpace[index])).all() else 0 for index in range(len(self.actionSpace))])
        return oneHotAction


class ApproximatePolicy:
    def __init__(self, model, actionSpace):
        self.actionSpace = actionSpace
        self.model = model

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
        graph = self.model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionIndices_ = graph.get_collection_ref("actionIndices")[0]
        actionIndices = self.model.run(
            actionIndices_, feed_dict={state_: stateBatch})
        actionBatch = [self.actionSpace[i] for i in actionIndices]
        if len(actionBatch) == 1:
            actionBatch = actionBatch[0]
        return actionBatch


class GetMcts():
    def __init__(self, actionSpace, numSimulations, selectChild, isTerminal, transit, terminalReward):
        self.numSimulations = numSimulations
        self.selectChild = selectChild
        self.actionSpace = actionSpace
        self.isTerminal = isTerminal
        self.transit = transit
        self.numActionSpace = len(actionSpace)
        self.terminalReward = terminalReward

    def __call__(self, NNPolicy):
        transitInMCTS = lambda state, action: self.transit(state, [action, NNPolicy(state)])
        getApproximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, self.actionSpace)
        getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(self.actionSpace, transitInMCTS,
                                                                          getApproximateActionPrior(NNModel))

        getExpandNNPrior = lambda NNModel: Expand(self.isTerminal, getInitializeChildrenNNPrior(NNModel))
        getStateFromNode = lambda node: list(node.id.values())[0]
        getEstimateValue = lambda NNModel: EstimateValueFromNode(self.terminalReward, self.isTerminal, getStateFromNode, ApproximateValueFunction(NNModel))

        getMCTSNNPriorValue = lambda NNModel: MCTS(self.numSimulations, self.selectChild, getExpandNNPrior(NNModel),
                                                   getEstimateValue(NNModel), backup, establishPlainActionDist)

        return getMCTSNNPriorValue


def main():
    # manipulated parameters and other important parameters

    manipulatedVariables = OrderedDict()
    # manipulatedVariables['numTrajectoriesPerIteration'] = [1]
    # manipulatedVariables['miniBatchSize'] = [64]
    # manipulatedVariables['learningRate'] = [0.001]
    # manipulatedVariables['bufferSize'] = [2000]

    # levelNames = list(manipulatedVariables.keys())
    # levelValues = list(manipulatedVariables.values())
    # modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    # toSplitFrame = pd.DataFrame(index=modelIndex)

    numTrajectoriesPerIteration = 1
    miniBatchSize = 64
    learningRate = 0.001
    bufferSize = 2000

    learningThresholdFactor = 4
    numIterations = 2000  # 2000
    numSimulations = 10  # 200

    maxRunningSteps = 30
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}

    # pre-process the trajectory for training the neural network
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 1 / maxRunningSteps
    playDeathPenalty = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]
    numActionSpace = len(sheepActionSpace)

    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    wolfActionToOneHot = ActionToOneHot(wolfActionSpace)
    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)

    processSheepTrajectoryForNN = ProcessTrajectoryForNN(sheepActionToOneHot, sheepId)
    preProcessSheepTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                         processSheepTrajectoryForNN)

    processWolfTrajectoryForNN = ProcessTrajectoryForNN(wolfActionToOneHot, wolfId)
    preProcessWolfTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                        processWolfTrajectoryForNN)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 1
    qPosInitNoise = 9.7

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # MCTS init
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    terminalReward = 1
    terminalPenalty = -1

    getSheepMCTS = GetMcts(sheepActionSpace, numSimulations, selectChild, isTerminal, transit, terminalReward)

    getWolfMCTS = GetMcts(wolfActionSpace, numSimulations, selectChild, isTerminal, transit, terminalPenalty)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getNNModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # replay buffer
    getUniformSamplingProbabilities = lambda buffer: [(1 / len(buffer)) for _ in buffer]
    getSampleBatchFromBuffer = lambda miniBatchSize: SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # function to train NN model
    batchSizeForTrainFunction = 0
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
    numTrainStepsPerIteration = 1
    trainReporter = TrainReporter(numTrainStepsPerIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda learningRate: Train(numTrainStepsPerIteration, batchSizeForTrainFunction, sampleData,
                                            learningRateModifier(learningRate),
                                            terminalController, coefficientController,
                                            trainReporter)

    trainNN = getTrainNN(learningRate)
    # functions to iteratively play and train the NN
    combineDict = lambda dict1, dict2: dict(
        list(dict1.items()) + list(dict2.items()))

    generatePathParametersAtIteration = lambda iterationIndex: \
        combineDict(NNFixedParameters,
                    {'iteration': iterationIndex})


# load save dir
    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesForSheepTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                          'trajectoriesForSheepTrain')
    if not os.path.exists(trajectoriesForSheepTrainSaveDirectory):
        os.makedirs(trajectoriesForSheepTrainSaveDirectory)

    sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'SheepWolfIterationPolicy', 'sheepPolicy')
    if not os.path.exists(sheepNNModelSaveDirectory):
        os.makedirs(sheepNNModelSaveDirectory)

    trajectoriesForWolfTrainSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                                         'trajectoriesForWolfTrain')
    if not os.path.exists(trajectoriesForWolfTrainSaveDirectory):
        os.makedirs(trajectoriesForWolfTrainSaveDirectory)

    wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                            'WolfWolfIterationPolicy', 'wolfPolicy')
    if not os.path.exists(wolfNNModelSaveDirectory):
        os.makedirs(wolfNNModelSaveDirectory)


# load wolf baseline for init iteration
    # wolfBaselineNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
    #                                                 'SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')
    # baselineSaveParameters = {'numSimulations': 10, 'killzoneRadius': 2,
    #                           'qPosInitNoise': 9.7, 'qVelInitNoise': 8,
    #                           'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 25}
    # getWolfBaselineModelSavePath = GetSavePath(wolfBaselineNNModelSaveDirectory, NNModelSaveExtension, baselineSaveParameters)
    # baselineModelTrainSteps = 1000
    # wolfBaselineNNModelSavePath = getWolfBaselineModelSavePath({'trainSteps': baselineModelTrainSteps})
    # wolfBaselienModel = restoreVariables(initializedNNModel, wolfBaselineNNModelSavePath)
    # wolfNNModel = wolfBaselienModel

    sampleBatchFromBuffer = getSampleBatchFromBuffer(miniBatchSize)
    saveToBuffer = SaveToBuffer(bufferSize)
    sheepBuffer = []
    wolfBuffer = []

    wolfNNModel = getNNModel()
    sheepNNModel = getNNModel()

    startTime = time.time()
    for iterationIndex in range(numIterations):
        print("ITERATION INDEX: ", iterationIndex)
        pathParametersAtIteration = generatePathParametersAtIteration(iterationIndex)

# sheep play
        # approximateWolfPolicy = ApproximatePolicy(wolfNNModel, wolfActionSpace)

        # getSheepPolicy = getSheepMCTS(approximateWolfPolicy)
        # sheepPolicy = getSheepPolicy(sheepNNModel)
        # wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}
        # policyForSheepTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        # trajectoriesForSheepTrain = [sampleTrajectory(policyForSheepTrain) for _ in range(numTrajectoriesPerIteration)]
        # getSheepTrajectorySavePath = GetSavePath(
        #     trajectoriesForSheepTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)

        # sheepDataSetPath = getSheepTrajectorySavePath(NNFixedParameters)
        # saveToPickle(trajectoriesForSheepTrain, sheepDataSetPath)

        # # sheepDataSetTrajectories = loadFromPickle(sheepDataSetPath)
        # sheepDataSetTrajectories = trajectoriesForSheepTrain

        # processedSheepTrajectories = preProcessSheepTrajectories(sheepDataSetTrajectories)
        # updatedSheepBuffer = saveToBuffer(sheepBuffer, processedSheepTrajectories)
        # if len(updatedSheepBuffer) >= learningThresholdFactor * miniBatchSize:
        #     sheepSampledBatch = sampleBatchFromBuffer(updatedBuffer)
        #     sheepTrainData = [list(varBatch) for varBatch in zip(*sheepSampledBatch)]
        #     updatedSheepNNModel = trainNN(sheepNNModel, sheepTrainData)
        #     sheepNNModel = updatedSheepNNModel

        # sheepBuffer = updatedSheepBuffer

        # # sheep train

        # getSheepModelSavePath = GetSavePath(
        #     sheepNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        # sheepNNModelSavePaths = getSheepModelSavePath({'killzoneRadius': killzoneRadius})
        # savedVariablesSheep = saveVariables(sheepNNModel, sheepNNModelSavePaths)


# wolf play
        approximateSheepPolicy = lambda state: (0, 0)

        # approximateSheepPolicy = ApproximatePolicy(sheepNNModel, sheepActionSpace)
        getWolfPolicy = getWolfMCTS(approximateSheepPolicy)
        wolfPolicy = getWolfPolicy(wolfNNModel)
        sheepPolicy = lambda state: {approximateSheepPolicy(state): 1}
        policyForWolfTrain = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        trajectoriesForWolfTrain = [sampleTrajectory(policyForWolfTrain) for _ in range(numTrajectoriesPerIteration)]

        getWolfTrajectorySavePath = GetSavePath(
            trajectoriesForWolfTrainSaveDirectory, trajectorySaveExtension, NNFixedParameters)
        wolfDataSetPath = getWolfTrajectorySavePath(NNFixedParameters)
        saveToPickle(trajectoriesForWolfTrain, wolfDataSetPath)

        # wolfDataSetTrajectories = loadFromPickle(wolfDataSetPath)
        wolfDataSetTrajectories = trajectoriesForWolfTrain

        processedWolfTrajectories = preProcessWolfTrajectories(wolfDataSetTrajectories)
        updatedWolfBuffer = saveToBuffer(wolfBuffer, processedWolfTrajectories)
        if len(updatedWolfBuffer) >= learningThresholdFactor * miniBatchSize:
            wolfSampledBatch = sampleBatchFromBuffer(updatedBuffer)
            wolfTrainData = [list(varBatch) for varBatch in zip(*wolfSampledBatch)]
            updatedWolfNNModel = trainNN(wolfNNModel, wolfTrainData)
            wolfNNModel = updatedWolfNNModel

        wolfBuffer = updatedWolfBuffer
# wolf train

        getWolfModelSavePath = GetSavePath(
            wolfNNModelSaveDirectory, NNModelSaveExtension, pathParametersAtIteration)
        wolfNNModelSavePaths = getWolfModelSavePath({'killzoneRadius': killzoneRadius})
        savedVariablesWolf = saveVariables(wolfNNModel, wolfNNModelSavePaths)

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        self.numIterations, (endTime - startTime)))


if __name__ == '__main__':
    main()
