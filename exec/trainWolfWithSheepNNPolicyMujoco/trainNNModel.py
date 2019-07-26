import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from src.neuralNetwork.policyValueNet import GenerateModel, Train, sampleData, saveVariables
from src.neuralNetwork.trainTools import TrainReporter, TrainTerminalController, CoefficientCotroller, \
    LearningRateModifier

import numpy as np


class ProcessTrajectoryForNN:
    def __init__(self, actionToOneHot, agentId):
        self.actionToOneHot = actionToOneHot
        self.agentId = agentId

    def __call__(self, trajectory):
        processTuple = lambda state, actions, actionDist, value: \
            (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        processedTrajectory = [processTuple(*tup) for tup in trajectory]

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
        allDataPoints = [dataPoint for trajectory in processedTrajectories for dataPoint in trajectory]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]

        return trainData


def main():
    # hyper parameters
    trainStepsEachIteration = 1
    learningRate = 0.0001
    miniBatchSize = 256
    numTrainIterations = 200000
    trainCheckPointInterval = 200

    # get trajectory save path
    trainDataNumSimulations = 125
    trainDataKillzoneRadius = 2
    trajectoryFixedParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius}
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'trainWolfWithSheepNNPolicyMujoco', 'trainingData')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)

    # load trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectories = loadTrajectories({})

    # pre-process trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = -0.05
    playDeathBonus = 1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getWolfXPos, getSheepXPos)
    playReward = RewardFunctionCompete(playAlivePenalty, playDeathBonus, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionToOneHot = lambda action: np.asarray([1 if (np.array(action) == np.array(actionSpace[index])).all() else 0
                                                for index in range(len(actionSpace))])
    actionIndex = 1
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForNN(actionToOneHot, wolfId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

    trainData = preProcessTrajectories(trajectories)

    # define NN Model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    NNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # path for saving NN
    NNModelSaveParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius,
                             'learningRate': learningRate, 'miniBatchSize': miniBatchSize}
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'trainWolfWithSheepNNPolicyMujoco', 'trainedNNModels2000TrainTrajectories')
    NNModelExtension = ''
    getNNModelPath = GetSavePath(NNModelSaveDirectory, NNModelExtension, NNModelSaveParameters)

    # train model
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
    reportInterval = 25
    trainReporter = TrainReporter(trainStepsEachIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    trainNN = Train(trainStepsEachIteration, miniBatchSize, sampleData, learningRateModifier,
                    terminalController, coefficientController, trainReporter)

    for iteration in range(numTrainIterations):
        updatedNNModel = trainNN(NNModel, trainData)
        NNModel = updatedNNModel
        if iteration % trainCheckPointInterval == 0 or iteration == numTrainIterations - 1:
            modelSavePath = getNNModelPath({'trainSteps': iteration})
            saveVariables(NNModel, modelSavePath)


if __name__ == '__main__':
    main()