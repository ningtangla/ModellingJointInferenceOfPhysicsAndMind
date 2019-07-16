import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

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
    def __init__(self, maxRunningSteps, restoreNNModel, allNNValuesForTrajectory):
        self.maxRunningSteps = maxRunningSteps
        self.restoreNNModel = restoreNNModel
        self.allNNValuesForTrajectory = allNNValuesForTrajectory

    def __call__(self, trainSteps):
        self.restoreNNModel(self.maxRunningSteps, trainSteps)
        return self.allNNValuesForTrajectory


class ComputeValueStatistics:
    def __init__(self, evaluationTrajectories):
        self.evaluationTrajectories = evaluationTrajectories

    def __call__(self, valueEstimator):
        allTrajectoriesValues = [valueEstimator(trajectory) for trajectory in self.evaluationTrajectories]
        allValues = [value for trajectoryValue in allTrajectoriesValues for value in trajectoryValue]
        meanValue = np.mean(allValues)
        valueStd = np.std(allValues)

        return meanValue, valueStd


class EvaluateValueEstimation:
    def __init__(self, getValueEstimator, computeValueStatistics):
        self.getValueEstimator = getValueEstimator
        self.computeValueStatistics = computeValueStatistics

    def __call__(self, oneConditionDf):
        trainSteps = oneConditionDf.index.get_level_values('iteration')[0]
        estimatorName = oneConditionDf.index.get_level_values('estimatorName')[0]
        valueEstimator = self.getValueEstimator(estimatorName, trainSteps)
        meanValueEstimate, valueEstimateStd = self.computeValueStatistics(valueEstimator)
        valueEstimateSeries = pd.Series({'mean': meanValueEstimate, 'std': valueEstimateStd})

        return valueEstimateSeries


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['estimatorName'] = ['trueValue', '10', '100']
    manipulatedVariables['iteration'] = [0, 200, 400, 600, 800, 999]
    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # load evaulation trajectories
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'evaluateEffectOfMaxRunningStepsOnValueNetWolfChaseSheepUsingMCTSMujoco',
                                           'valueEstimationEvaluationTrajectories')
    trajectorySaveParameters = {'numSimulations': 75, 'killzoneRadius': 0.5, 'qPosInitNoise': 9.7, 'qVelInitNoise': 1,
                                'rolloutHeuristicWeight': 0.1, 'maxRunningSteps': 100}
    trajectorySaveExtension = '.pickle'
    getEvaluationTrajectoriesPath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    loadTrajectories = LoadTrajectories(getEvaluationTrajectoriesPath, loadFromPickle)

    evaluationTrajectories = loadTrajectories({})

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
    NNFixedParameters = {'numSimulations': 75, 'killzoneRadius': 0.5, 'qPosInitNoise': 9.7, 'qVelInitNoise': 1,
                         'rolloutHeuristicWeight': 0.1}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'evaluateEffectOfMaxRunningStepsOnValueNetWolfChaseSheepUsingMCTSMujoco',
                                        'trainedNNModels')
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
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getWolfXPos, getSheepXPos)
    playReward = RewardFunctionCompete(playAlivePenalty, playDeathBonus, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)
    valueIndex = -1
    allTrueValuesForTrajectory = lambda trajectory: [timeStep[valueIndex] for timeStep in
                                                     addValuesToTrajectory(trajectory)]
    trueValueEstimator = lambda trainSteps: allTrueValuesForTrajectory

    # value estimators--Neural nets
    approximateValueFunction = ApproximateValueFunction(initializedNNModel)
    getFlattenedStateFromTrajectory = lambda trajectory, timeStep: trajectory[timeStep][0].flatten()
    allNNValuesForTrajectory = lambda trajectory: \
        [approximateValueFunction(getFlattenedStateFromTrajectory(trajectory, timeStep)) for timeStep in range(len(trajectory))]

    restoreNNModel = lambda estimatorName, trainSteps: \
        restoreVariables(initializedNNModel, getNNModelSavePath({'iteration': trainSteps, 'maxRunningSteps': estimatorName}))
    getNNValueEstimator = lambda maxRunningSteps: NNValueEstimator(maxRunningSteps, restoreNNModel, allNNValuesForTrajectory)

    # value estimator wrapper
    allValueEstimators = {'trueValue': trueValueEstimator, '10': getNNValueEstimator(10), '100': getNNValueEstimator(100)}
    getValueEstimator = lambda estimatorName, trainSteps: allValueEstimators[estimatorName](trainSteps)

    # compute mean value
    computeValueStatistics = ComputeValueStatistics(evaluationTrajectories)

    # evaluate value estimation
    evaluateValueEstimation = EvaluateValueEstimation(getValueEstimator, computeValueStatistics)
    levelNames = list(manipulatedVariables.keys())
    evaluateDf = toSplitFrame.groupby(levelNames).apply(evaluateValueEstimation)

    fig = plt.figure()
    axForDraw = fig.add_subplot(1, 1, 1)

    for estimator, grp in evaluateDf.groupby('estimatorName'):
        grp.index = grp.index.droplevel('estimatorName')
        grp.plot(marker='o', label=estimator, ax=axForDraw, y='mean', yerr='std')

    plt.title('estimation of value as a function of training steps')
    plt.legend(loc='best')
    plt.ylabel('value estimate')
    plt.show()


if __name__ == '__main__':
    main()