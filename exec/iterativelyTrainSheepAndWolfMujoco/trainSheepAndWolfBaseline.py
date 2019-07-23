import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, loadFromPickle
from src.neuralNetwork.policyNet import GenerateModel, Train, saveVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, \
    LearningRateModifier
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory
from exec.trajectoriesSaveLoad import conditionDfFromParametersDict


import numpy as np
import random
from collections import OrderedDict


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(
            self.actionSpace[index])).all() else 0 for index in range(len(self.actionSpace))])
        return oneHotAction


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot

    def __call__(self, trajectories):
        stateActionPairs = [
            pair for trajectory in trajectories for pair in trajectory]
        stateActionPairsFiltered = list(filter(
            lambda pair: pair[self.actionIndex] is not None and pair[0][1][2] < 9.7, stateActionPairs))
        print("{} data points remain after filtering".format(
            len(stateActionPairsFiltered)))
        stateActionPairsProcessed = [(np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]))
                                     for state, actions, actionDist in stateActionPairsFiltered]

        return stateActionPairsProcessed


def main():
    # get trajectory save path
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'generateTrajectoryForSheepAndWolfBaseline')
    numSimulations = 10
    maxRunningSteps = 25
    killzoneRadius = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 8
    rolloutHeuristicWeight = 0.1
    trajectorySaveParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                                'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                                'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(
        trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    dataSetPath = getTrajectorySavePath(trajectorySaveParameters)
    dataSetTrajectories = loadFromPickle(dataSetPath)

    # pre process trajectories
    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                        (-10, 0), (-7, -7), (0, -10), (7, -7)]
    wolfActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6),
                       (-8, 0), (-6, -6), (0, -8), (6, -6)]
    numActionSpace = len(sheepActionSpace)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAlivePenalty = 0.05
    playDeathBonus = -1
    playKillzoneRadius = 2
    playIsTerminal = IsTerminal(playKillzoneRadius, getWolfXPos, getSheepXPos)
    playReward = RewardFunctionCompete(
        playAlivePenalty, playDeathBonus, playIsTerminal)

    actionIndex = 1
# prepare sheep training data
    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)
    preProcessSheepTrajectories = PreProcessTrajectories(
        sheepId, actionIndex, sheepActionToOneHot)
    sheepStateActionPairsProcessed = preProcessSheepTrajectories(
        dataSetTrajectories)
    random.shuffle(sheepStateActionPairsProcessed)
    sheepTrainData = [[state for state, action in sheepStateActionPairsProcessed],
                      [action for state, action in sheepStateActionPairsProcessed]]

# prepare wolf training data
    wolfActionToOneHot = ActionToOneHot(wolfActionSpace)
    preProcessWolfTrajectories = PreProcessTrajectories(
        wolfId, actionIndex, sheepActionToOneHot)
    wolfStateActionPairsProcessed = preProcessWolfTrajectories(
        dataSetTrajectories)

    random.shuffle(wolfStateActionPairsProcessed)
    wolfTrainData = [[state for state, action in wolfStateActionPairsProcessed],
                     [action for state, action in wolfStateActionPairsProcessed]]

    # initialise model for training
    numStateSpace = 12
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModel(
        numStateSpace, numActionSpace, learningRate, regularizationFactor)

    # train models
    allTrainSteps = [0, 1000]  # [1000, 5000, 10000, 15000]
    reportInterval = 500
    lossChangeThreshold = 1e-6
    lossHistorySize = 10
    getTrain = lambda trainSteps: Train(
        trainSteps, learningRate, lossChangeThreshold, lossHistorySize, reportInterval, summaryOn=False, testData=None)
    allTrainFunctions = {trainSteps: getTrain(
        trainSteps) for trainSteps in allTrainSteps}

    NNFixedParameters = {'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,
                         'qPosInitNoise': qPosInitNoise, 'qVelInitNoise': qVelInitNoise,
                         'rolloutHeuristicWeight': rolloutHeuristicWeight, 'maxRunningSteps': maxRunningSteps}
    dirName = os.path.dirname(__file__)

# train sheep
    allTrainedSheepModels = {trainSteps: train(generatePolicyNet(
        hiddenWidths), sheepTrainData) for trainSteps, train in allTrainFunctions.items()}

    # NN save path
    sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'SheepWolfBaselinePolicy', 'sheepBaselineNNPolicy')
    if not os.path.exists(sheepNNModelSaveDirectory):
        os.makedirs(sheepNNModelSaveDirectory)
    NNModelSaveExtension = ''

    getSheepModelSavePath = GetSavePath(
        sheepNNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    allSheepModelSavePaths = {trainedModel: getSheepModelSavePath(
        {'trainSteps': trainSteps}) for trainSteps, trainedModel in allTrainedSheepModels.items()}

    # save trained model variables
    savedVariablesSheep = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                           allSheepModelSavePaths.items()]

# train wolf
    allTrainedWolfModels = {trainSteps: train(generatePolicyNet(hiddenWidths), wolfTrainData) for trainSteps, train in
                            allTrainFunctions.items()}
    wolfNNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                            'SheepWolfBaselinePolicy', 'wolfBaselineNNPolicy')

    if not os.path.exists(wolfNNModelSaveDirectory):
        os.makedirs(wolfNNModelSaveDirectory)
    getWolfModelSavePath = GetSavePath(
        wolfNNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    allWolfModelSavePaths = {trainedModel: getWolfModelSavePath(
        {'trainSteps': trainSteps}) for trainSteps, trainedModel in allTrainedSheepModels.items()}

    savedVariablesWolf = [saveVariables(trainedModel, modelSavePath) for trainedModel, modelSavePath in
                          allWolfModelSavePaths.items()]


if __name__ == '__main__':
    main()
