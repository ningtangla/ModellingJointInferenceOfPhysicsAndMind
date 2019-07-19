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


def main():
    # manipulated parameters and other important parameters
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrajectoriesPerIteration'] = [1]
    manipulatedVariables['miniBatchSize'] = [64]
    manipulatedVariables['learningRate'] = [0.001]
    manipulatedVariables['bufferSize'] = [2000]
    learningThresholdFactor = 4
    numIterations = 1000
    numSimulations = 150

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getNNModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    maxRunningSteps = 15
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                        'replayBufferEscape', 'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    saveNNModel = lambda NNmodel, pathParameters: saveVariables(NNmodel, getNNModelSavePath(pathParameters))

    # trajectory path to load
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                           'replayBufferEscape', 'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # load trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

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

    actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
                                                for index in range(len(actionSpace))])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForNN(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

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

    reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # minIterLinearRegion = 200
    # maxIterLinearRegion = 5000
    # stdDevMin = 2
    # stdDevMax = 16
    # getQPosInitStdDev = QPosInitStdDevForIteration(minIterLinearRegion, stdDevMin, maxIterLinearRegion, stdDevMax)
    # qMin = (-9.7, -9.7)
    # qMax = (9.7, 9.7)
    # withinBounds = WithinBounds(qMin, qMax)
    # getResetFromQPosStdDev = lambda qPosInitStdDev: \
    #     ResetGaussian(physicsSimulation, qVelInit, numAgents, qPosInitStdDev, qVelInitNoise, qMin, qMax, withinBounds)

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    wolfActionInSheepMCTSSimulation = lambda state: (0, 0)
    transitInSheepMCTSSimulation = lambda state, sheepSelfAction: \
        transit(state, [sheepSelfAction, wolfActionInSheepMCTSSimulation(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # functions to make predictions from NN
    getApproximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                                      getApproximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))

    getStateFromNode = lambda node: list(node.id.values())[0]
    terminalPenalty = -1
    getEstimateValue = lambda NNModel: \
        EstimateValueFromNode(terminalPenalty, isTerminal, getStateFromNode, ApproximateValueFunction(NNModel))

    # wrapper for MCTS
    getMCTSNNPriorValue = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                               getEstimateValue(NNModel), backup, establishPlainActionDist)

    # heatseeking wolf policy
    wolfActionMagnitude = 5
    heatSeekingWolfPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfXPos, getSheepXPos, wolfActionMagnitude)
    getWolfPolicy = lambda NNModel: heatSeekingWolfPolicy

    # policy
    preparePolicy = PreparePolicy(getWolfPolicy, getMCTSNNPriorValue)

    # sample trajectory
    # getSampleTrajectoryFromReset = lambda reset: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset,
    #                                                               chooseGreedyAction)
    # getSampleTrajectoryForIteration = GetSampleTrajectoryForIteration(getQPosInitStdDev, getResetFromQPosStdDev,
    #                                                                   getSampleTrajectoryFromReset)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # generate trajectories
    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)
    # getGenerateTrajectories = lambda numTrajectoriesPerIteration: GenerateTrajectories(numTrajectoriesPerIteration,
    #                                                                                    getSampleTrajectoryForIteration,
    #                                                                                    preparePolicy, saveAllTrajectories)
    getGenerateTrajectories = lambda numTrajectoriesPerIteration: GenerateTrajectories(numTrajectoriesPerIteration,
                                                                                       sampleTrajectory, preparePolicy,
                                                                                       saveAllTrajectories)

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

    # functions to iteratively play and train the NN
    combineDict = lambda dict1, dict2: dict(list(dict1.items()) + list(dict2.items()))
    generatePathParametersAtIteration = lambda oneConditionDf, iterationIndex: \
        combineDict(readParametersFromDf(oneConditionDf), {'iteration': iterationIndex})
    iterativePlayAndTrain = IterativePlayAndTrain(numIterations, learningThresholdFactor, saveNNModel,
                                                  getGenerateTrajectories, preProcessTrajectories,
                                                  getSampleBatchFromBuffer, getTrainNN, getNNModel, loadTrajectories,
                                                  SaveToBuffer, generatePathParametersAtIteration, getNNModelSavePath,
                                                  restoreVariables)

    performanceDf = toSplitFrame.groupby(levelNames).apply(iterativePlayAndTrain)


if __name__ == '__main__':
    main()
