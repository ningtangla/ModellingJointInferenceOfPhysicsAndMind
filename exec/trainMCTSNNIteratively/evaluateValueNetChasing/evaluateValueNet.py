import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from exec.trajectoriesSaveLoad import LoadTrajectories, loadFromPickle, GetSavePath
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateValueFunction, restoreVariables
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.evaluationFunctions import conditionDfFromParametersDict

import pandas as pd
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt


class NNValueEstimator:
    def __init__(self, restoreNNModel, allNNValuesForTrajectory):
        self.restoreNNModel = restoreNNModel
        self.allNNValuesForTrajectory = allNNValuesForTrajectory

    def __call__(self, iteration):
        self.restoreNNModel(iteration)
        return self.allNNValuesForTrajectory


class ComputeValueStatistics:
    def __init__(self, evaluationTrajectories, computeTrueValues):
        self.evaluationTrajectories = evaluationTrajectories
        self.computeTrueValues = computeTrueValues

    def __call__(self, valueEstimator):
        allTrajectoriesEstimatedValues = [valueEstimator(trajectory) for trajectory in self.evaluationTrajectories]
        allEstimatedValues = np.array([value for trajectoryValue in allTrajectoriesEstimatedValues for value in trajectoryValue]).flatten()

        allTrajectoriesTrueValues = [self.computeTrueValues(trajectory) for trajectory in self.evaluationTrajectories]
        allTrueValues = np.array([value for trajectoryValue in allTrajectoriesTrueValues for value in trajectoryValue]).flatten()

        valueEstimationError = allEstimatedValues-allTrueValues

        meanErrorL2Norm = np.linalg.norm((valueEstimationError), 1)/np.size(valueEstimationError)

        return meanErrorL2Norm


class EvaluateValueEstimation:
    def __init__(self, getValueEstimator, computeValueStatistics):
        self.getValueEstimator = getValueEstimator
        self.computeValueStatistics = computeValueStatistics

    def __call__(self, oneConditionDf):
        iteration = oneConditionDf.index.get_level_values('iteration')[0]
        estimatorName = oneConditionDf.index.get_level_values('estimatorName')[0]
        valueEstimator = self.getValueEstimator(estimatorName, iteration)
        meanErrorL2Norm = self.computeValueStatistics(valueEstimator)
        valueEstimateSeries = pd.Series({'meanErrorL2Norm': meanErrorL2Norm})

        return valueEstimateSeries


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['estimatorName'] = ['NNValue']
    manipulatedVariables['iteration'] = [0, 500, 2000, 8000, 16000, 19999]
    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # load evaulation trajectories
    evaluationTrajectoriesPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/evaluateValueNetChasing/evaluationTrajectories/maxRunningSteps=20_policyName=NNIterative20kTrainSteps.pickle'
    evaluationTrajectories = loadFromPickle(evaluationTrajectoriesPath)
    evaluationTrajectories = evaluationTrajectories[0:200]

    # NN Model
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # path to load NN Model
    trainBufferSize = 2000
    trainLearningRate = 0.0001
    trainMaxRunningSteps = 20
    trainMiniBatchSize = 256
    trainNumSimulations = 200
    trainNumTrajectoriesPerIteration = 1
    NNFixedParameters = {'bufferSize': trainBufferSize, 'learningRate': trainLearningRate,
                         'maxRunningSteps': trainMaxRunningSteps, 'miniBatchSize': trainMiniBatchSize,
                         'numSimulations': trainNumSimulations, 'numTrajectoriesPerIteration': trainNumTrajectoriesPerIteration}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data',
                                        'trainMCTSNNIteratively', 'replayBufferStartWithTrainedModel', 'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # value estimators--true value
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
    valueIndex = -1
    allTrueValuesForTrajectory = lambda trajectory: [timeStep[valueIndex] for timeStep in
                                                     addValuesToTrajectory(trajectory)]

    # value estimators--Neural nets
    approximateValueFunction = ApproximateValueFunction(initializedNNModel)
    getFlattenedStateFromTrajectory = lambda trajectory, timeStep: trajectory[timeStep][0].flatten()
    allNNValuesForTrajectory = lambda trajectory: \
        [approximateValueFunction(getFlattenedStateFromTrajectory(trajectory, timeStep)) for timeStep in range(len(trajectory))]

    restoreNNModel = lambda iteration: restoreVariables(initializedNNModel, getNNModelSavePath({'iteration': iteration}))
    nnValueEstimator = NNValueEstimator(restoreNNModel, allNNValuesForTrajectory)

    # value estimator wrapper
    allValueEstimators = {'NNValue': nnValueEstimator}
    getValueEstimator = lambda estimatorName, iteration: allValueEstimators[estimatorName](iteration)

    # compute mean value
    computeValueStatistics = ComputeValueStatistics(evaluationTrajectories, allTrueValuesForTrajectory)

    # evaluate value estimation
    evaluateValueEstimation = EvaluateValueEstimation(getValueEstimator, computeValueStatistics)
    levelNames = list(manipulatedVariables.keys())
    evaluateDf = toSplitFrame.groupby(levelNames).apply(evaluateValueEstimation)

    fig = plt.figure()
    axForDraw = fig.add_subplot(1, 1, 1)

    for estimator, grp in evaluateDf.groupby('estimatorName'):
        grp.index = grp.index.droplevel('estimatorName')
        grp.plot(marker='o', label=estimator, ax=axForDraw, y='meanErrorL2Norm')

    plt.title('estimation of value as a function of training iterations\nIterative training (bootstrapping). Start with pre-trained Model.')
    plt.legend(loc='best')
    plt.ylabel('Mean L2 Norm of value estimation error')
    plt.show()


if __name__ == '__main__':
    main()