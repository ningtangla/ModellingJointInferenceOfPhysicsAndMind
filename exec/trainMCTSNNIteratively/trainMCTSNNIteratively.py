import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import OrderedDict
from mujoco_py import load_model_from_path, MjSim

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from src.play import SampleTrajectory, agentDistToGreedyAction, worldDistToAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateActionPrior, ApproximateValueFunction, \
    Train, saveVariables, sampleData
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientController, TrainTerminalController, TrainReporter
from exec.trainMCTSNNIteratively.wrappers import GetApproximateValueFromNode, getStateFromNode
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer


def saveToPickle(data, path):
    pickleOut = open(path, 'wb')
    pickle.dump(data, pickleOut)
    pickleOut.close()


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


class GetPolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, NNModel):
        sheepPolicy = self.getSheepPolicy(NNModel)
        wolfPolicy = self.getWolfPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class PlayToTrain:
    def __init__(self, numTrialsEachIteration, sampleTrajectory, getTrajectorySavePath, getPolicy, saveData):
        self.numTrialsEachIteration = numTrialsEachIteration
        self.sampleTrajectory = sampleTrajectory
        self.getTrajectorySavePath = getTrajectorySavePath
        self.getPolicy = getPolicy
        self.saveData = saveData

    def __call__(self, NNModel, pathParameters):
        policy = self.getPolicy(NNModel)
        trajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrialsEachIteration)]
        trajectorySavePath = self.getTrajectorySavePath(pathParameters)
        self.saveData(trajectories, trajectorySavePath)

        return trajectories


class ConstantLearningRateModifier:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def __call__(self, trainIteration):
        return self.learningRate


class TrainToPlay:
    def __init__(self, train, getModelSavePath):
        self.train = train
        self.getModelSavePath = getModelSavePath

    def __call__(self, NNModel, trainData, pathParameters):
        updatedNNModel = self.train(NNModel, trainData)
        modelSavePath = self.getModelSavePath(pathParameters)
        saveVariables(updatedNNModel, modelSavePath)

        return updatedNNModel


class IterativePlayAndTrain:
    def __init__(self, numTrainStepsPerIteration, getPlayToTrain, getTrainToPlay, preProcessTrajectories, saveToBuffer,
                 getSampleBatchFromBuffer, getModel):
        self.numTrainStepsPerIteration = numTrainStepsPerIteration
        self.getPlayToTrain = getPlayToTrain
        self.getTrainToPlay = getTrainToPlay
        self.preProcessTrajectories = preProcessTrajectories
        self.saveToBuffer = saveToBuffer
        self.getSampleBatchFromBuffer = getSampleBatchFromBuffer
        self.getModel = getModel

    def __call__(self, oneConditionDf):
        numTrialsPerIteration = oneConditionDf.index.get_level_values('numTrialsPerIteration')[0]
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]

        indexLevelNames = oneConditionDf.index.names
        pathParameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}

        NNModel = self.getModel()
        buffer = []
        playToTrain = self.getPlayToTrain(numTrialsPerIteration)
        trainToPlay = self.getTrainToPlay(learningRate)
        numIterations = trainSteps/self.numTrainStepsPerIteration

        for iteration in range(numIterations):
            trajectories = playToTrain(NNModel, numTrialsPerIteration, pathParameters)
            processedTrajectories = self.preProcessTrajectories(trajectories)
            updatedBuffer = [self.saveToBuffer(buffer, trajectory) for trajectory in processedTrajectories]
            sampleFromBuffer = self.getSampleBatchFromBuffer(miniBatchSize)
            trainData = [list(varBatch) for varBatch in zip(*sampleFromBuffer)]
            updatedNNModel = trainToPlay(NNModel, trainData, pathParameters)
            NNModel = updatedNNModel
            buffer = updatedBuffer


<<<<<<< HEAD
def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrialsPerIteration'] = []
    manipulatedVariables['miniBatchSize'] = []
    manipulatedVariables['learningRate'] = []

    # commonly varied parameters
    numIterations = 100
    numTrainStepsPerIteration = 1
    maxRunningSteps = 10
    numSimulations = 75

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    mujocoModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    mujocoModel = load_model_from_path(mujocoModelPath)
    simulation = MjSim(mujocoModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(simulation, isTerminal, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    # MCTS details
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # wrapper function for expand
    approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, sheepTransit, approximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))

    # wrapper function for policy
    gnumActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    # value in MCTSNN
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)etMCTSNN = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                     getApproximateValue(NNModel), backup, establishPlainActionDist)
    getStationaryAgentPolicy = lambda NNModel: stationaryAgentPolicy                                                    # should I do this just to keep the interface symmetric?
    getPolicy = GetPolicy(getMCTSNN, getStationaryAgentPolicy)

    # sample trajectory
    reset = Reset(simulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, distToAction)

    # pre-process the trajectory for training the neural network
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionIndex = 1
    actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
                                                for index in range(len(actionSpace))])

    # function to train NN model
    batchSizeForTrainFunction = 0
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientController(initActionCoeff, initValueCoeff)
    reportInterval = 25
    getTrainReporter = lambda trainSteps: TrainReporter(trainSteps, reportInterval)
    getTrain = lambda trainSteps, learningRate: Train(trainSteps, batchSizeForTrainFunction, sampleData,
                                                      ConstantLearningRateModifier(learningRate),
                                                      terminalController, coefficientController,
                                                      getTrainReporter(trainSteps))

    # NN model save path
    trainFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations}
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)

    # trajectory save path
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                           'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # replay buffer
    replayBuffer = []
    windowSize = 10000
    saveToBuffer = SaveToBuffer(windowSize)
    getUniformSamplingProbabilities = lambda buffer: [(1/len(buffer)) for _ in buffer]
    getSampleBatchFromBuffer = lambda miniBatchSize: SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # functions to iteratively play and train the NN
    getPlayToTrain = lambda numTrialsPerIteration: PlayToTrain(numTrialsPerIteration, sampleTrajectory,
                                                               getTrajectorySavePath, getPolicy, saveData)
    getTrainToPlay = lambda learningRate: TrainToPlay(getTrain(numTrainStepsPerIteration, learningRate), getModelSavePath)
    iterativePlayAndTrain = IterativePlayAndTrain(numTrainStepsPerIteration, getPlayToTrain, getTrainToPlay, preProcessTrajectories, saveToBuffer, getSampleBatchFromBuffer, getModel)

if __name__ == '__main__':
    main()
=======
# def main():
    # manipulatedVariables = OrderedDict()
    # manipulatedVariables['numTrialsPerIteration'] = []
    # manipulatedVariables['miniBatchSize'] = []
    # manipulatedVariables['learningRate'] = []
    #
    # # commonly varied parameters
    # numIterations = 100
    # numTrainStepsPerIteration = 1
    # maxRunningSteps = 10
    # numSimulations = 75
    #
    # # Mujoco environment
    # dirName = os.path.dirname(__file__)
    # mujocoModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    # mujocoModel = load_model_from_path(mujocoModelPath)
    # simulation = MjSim(mujocoModel)
    # qPosInit = (0, 0, 0, 0)
    # qVelInit = [0, 0, 0, 0]
    # numAgents = 2
    # qPosInitNoise = 9.7
    # qVelInitNoise = 0
    #
    # sheepId = 0
    # wolfId = 1
    # xPosIndex = [2, 3]
    # getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    # getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    #
    # killzoneRadius = 0.5
    # isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)
    #
    # numSimulationFrames = 20
    # transit = TransitionFunction(simulation, isTerminal, numSimulationFrames)
    # sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])
    #
    # # MCTS details
    # cInit = 1
    # cBase = 100
    # calculateScore = ScoreChild(cInit, cBase)
    # selectChild = SelectChild(calculateScore)
    #
    # actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    # numActionSpace = len(actionSpace)
    #
    # # neural network model
    # numStateSpace = 12
    # regularizationFactor = 1e-4
    # sharedWidths = [128]
    # actionLayerWidths = [128]
    # valueLayerWidths = [128]
    # generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    # getModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    #
    # # wrapper function for expand
    # approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    # getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, sheepTransit, approximateActionPrior(NNModel))
    # getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))
    #
    # # wrapper function for policy
    # getApproximateValue = lambda NNModel: GetApproximateValueFromNode(getStateFromNode, ApproximateValueFunction(NNModel))
    # getMCTSNN = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
    #                                  getApproximateValue(NNModel), backup, establishPlainActionDist)
    # getStationaryAgentPolicy = lambda NNModel: stationaryAgentPolicy                                                    # should I do this just to keep the interface symmetric?
    # getPolicy = GetPolicy(getMCTSNN, getStationaryAgentPolicy)
    #
    # # sample trajectory
    # reset = Reset(simulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    # distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, distToAction)
    #
    # # pre-process the trajectory for training the neural network
    # playAliveBonus = -0.05
    # playDeathPenalty = 1
    # playKillzoneRadius = 0.5
    # playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    # playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)
    #
    # decay = 1
    # accumulateRewards = AccumulateRewards(decay, playReward)
    # addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)
    #
    # actionIndex = 1
    # actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
    #                                             for index in range(len(actionSpace))])
    #
    # # function to train NN model
    # batchSizeForTrainFunction = 0
    # terminalThreshold = 1e-6
    # lossHistorySize = 10
    # initActionCoeff = 1
    # initValueCoeff = 1
    # terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    # coefficientController = CoefficientController(initActionCoeff, initValueCoeff)
    # reportInterval = 25
    # getTrainReporter = lambda trainSteps: TrainReporter(trainSteps, reportInterval)
    # getTrain = lambda trainSteps, learningRate: Train(trainSteps, batchSizeForTrainFunction, sampleData,
    #                                                   ConstantLearningRateModifier(learningRate),
    #                                                   terminalController, coefficientController,
    #                                                   getTrainReporter(trainSteps))
    #
    # # NN model save path
    # trainFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
    #                         'numSimulations': numSimulations}
    # NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
    #                                     'trainedNNModels')
    # if not os.path.exists(NNModelSaveDirectory):
    #     os.makedirs(NNModelSaveDirectory)
    # NNModelSaveExtension = ''
    # getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)
    #
    # # trajectory save path
    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
    #                              'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    # trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
    #                                        'trajectories')
    # if not os.path.exists(trajectorySaveDirectory):
    #     os.makedirs(trajectorySaveDirectory)
    # trajectoryExtension = '.pickle'
    # getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)
    #
    # # replay buffer
    # replayBuffer = []
    # windowSize = 10000
    # saveToBuffer = SaveToBuffer(windowSize)
    # getUniformSamplingProbabilities = lambda buffer: [(1/len(buffer)) for _ in buffer]
    # getSampleBatchFromBuffer = lambda miniBatchSize: SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)
    #
    # # functions to iteratively play and train the NN
    # getPlayToTrain = lambda numTrialsPerIteration: PlayToTrain(numTrialsPerIteration, sampleTrajectory,
    #                                                            getTrajectorySavePath, getPolicy, saveData)
    # getTrainToPlay = lambda learningRate: TrainToPlay(getTrain(numTrainStepsPerIteration, learningRate), getModelSavePath)
    # iterativePlayAndTrain = IterativePlayAndTrain(numTrainStepsPerIteration, getPlayToTrain, getTrainToPlay, preProcessTrajectories, saveToBuffer, getSampleBatchFromBuffer, getModel)


# if __name__ == '__main__':
#     main()
>>>>>>> mctsMujocoSingleAgent
