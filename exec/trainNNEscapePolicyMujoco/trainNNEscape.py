import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, \
    LearningRateModifier
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory

import numpy as np


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
        allDataPoints = [dataPoint for trajectory in processedTrajectories for dataPoint in trajectory]
        trainData = [list(varBatch) for varBatch in zip(*allDataPoints)]

        return trainData


class TrainNNModel:
    def __init__(self, numTrainIterations, trainCheckPointInterval, readParametersFromDf,
                 loadTrajectories, preProcessTrajectories, getModel, trainNNModel, getModelSavePath, saveVariables):
        self.numTrainIterations = numTrainIterations
        self.trainCheckPointInterval = trainCheckPointInterval
        self.readParametersFromDf = readParametersFromDf
        self.loadTrajectories = loadTrajectories
        self.preProcessTrajectories = preProcessTrajectories
        self.getModel = getModel
        self.trainNNModel = trainNNModel
        self.getModelSavePath = getModelSavePath
        self.saveVariables = saveVariables

    def __call__(self):
        trajectories = self.loadTrajectories({})
        trainData = self.preProcessTrajectories(trajectories)
        NNModel = self.getModel()
        for iteration in range(self.numTrainIterations):
            updatedNNModel = self.trainNNModel(NNModel, trainData)
            NNModel = updatedNNModel
            if iteration % self.trainCheckPointInterval == 0 or iteration == self.numTrainIterations - 1:
                pathParameters = {}
                pathParameters['trainSteps'] = iteration
                modelSavePath = self.getModelSavePath(pathParameters)
                self.saveVariables(NNModel, modelSavePath)


def main():
    trainStepsEachIteration = 1
    trainCheckpointInterval = 200
    numTrainIterations = 100000
    learningRate = 0.0001
    miniBatchSize = 64

    # get trajectory save path
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainNNEscapePolicyMujoco', 'trainingTrajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)

    numSimulations = 100
    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = -0.1
    maxRunningSteps = 25
    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)

    # load trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

    # pre process trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = 0.05
    playDeathBonus = -1
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
    processTrajectoryForNN = ProcessTrajectoryForNN(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

    # neural network model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getNNModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # NN save path
    NNFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                         'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                         'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainNNEscapePolicyMujoco', 'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # train NNModel
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

    # train model for condition
    trainNNModel = TrainNNModel(numTrainIterations, trainCheckpointInterval, readParametersFromDf,
                                loadTrajectories, preProcessTrajectories, getNNModel, trainNN, getNNModelSavePath,
                                saveVariables)

    trainNNModel()

if __name__ == '__main__':
    main()