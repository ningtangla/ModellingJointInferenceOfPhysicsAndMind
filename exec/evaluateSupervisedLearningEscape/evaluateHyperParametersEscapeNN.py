import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
from mujoco_py import load_model_from_path, MjSim

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


def drawPerformanceLine(dataDf, axForDraw, deth):
    for learningRate, grp in dataDf.groupby('learningRate'):
        grp.index = grp.index.droplevel('learningRate')
        grp.plot(ax=axForDraw, label='learningRate={}'.format(learningRate), y='actionLoss',
                 title='deth: {}'.format(deth), marker='o', logx=True)


class TrainModelForConditions:
    def __init__(self, trainData, getNNModel, getTrain, getModelSavePath):
        self.trainData = trainData
        self.getNNModel = getNNModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, oneConditionDf):
        miniBatchSize = oneConditionDf.index.get_level_values('miniBatchSize')[0]
        learningRate = oneConditionDf.index.get_level_values('learningRate')[0]
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        depth = oneConditionDf.index.get_level_values('depth')[0]

        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        modelSavePath = self.getModelSavePath(parameters)

        model = self.getNNModel(depth)

        if not os.path.isfile(modelSavePath):
            train = self.getTrain(trainSteps, miniBatchSize, learningRate)
            trainedModel = train(model, self.trainData)
            saveVariables(trainedModel, modelSavePath)

            graph = trainedModel.graph
            state_ = graph.get_collection_ref("inputs")[0]
            groundTruthAction_, groundTruthValue_ = graph.get_collection_ref("groundTruths")
            actionLoss_ = graph.get_collection_ref("actionLoss")[0]
            fetches = {"actionLoss": actionLoss_}

            stateBatch, actionBatch, valueBatch = self.trainData
            evalDict = model.run(fetches, feed_dict={state_: stateBatch, groundTruthAction_: actionBatch, groundTruthValue_: valueBatch})

        return pd.Series({'actionLoss': evalDict['actionLoss']})


def main():
    # important parameters
    evalNumTrials = 1000  # 200
    sheepId = 0

    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['miniBatchSize'] = [64, 128, 256]
    manipulatedVariables['learningRate'] = [1e-2, 1e-3, 1e-4]
    manipulatedVariables['trainSteps'] = [0, 1000, 2000, 3000]
    manipulatedVariables['depth'] = [1, 2, 3]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearningEscape',
                                    'trajectories')
    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 20
    dataSetNumSimulations = 100
    killzoneRadius = 2

    dataSetFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)
    playAliveBonus = 0.05
    playDeathPenalty = -1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)
    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory,
                                                    processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    trajectories = loadTrajectories(parameters={})
    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))
    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    getNNModel = lambda depth: generateModel(sharedWidths, actionLayerWidths * depth, valueLayerWidths)
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
    trainReporter = lambda trainSteps: TrainReporter(trainSteps, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda trainSteps, batchSize, learningRate, : Train(trainSteps, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController,
                                                                     trainReporter(trainSteps))

    # get path to save trained models
    NNModelFixedParameters = {'agentId': sheepId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearningEscape',
                                        'trainedModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    trainModelForConditions = TrainModelForConditions(trainData, getNNModel, getTrainNN, getNNModelSavePath)

    # train models for all conditions
    statisticsDf = toSplitFrame.groupby(levelNames).apply(trainModelForConditions)

    # # all agent policies
    # getApproximatePolicy = lambda NNModel: ApproximatePolicy(NNModel, actionSpace)
    # actionMagnitude = 5
    # heatSeekingDeterministicPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfPos, getSheepPos, actionMagnitude)
    # getWolfPolicy = lambda NNModel: heatSeekingDeterministicPolicy
    # getPolicy = GetPolicy(getApproximatePolicy, getWolfPolicy)

    # # sample trajectory for evaluation
    # evalMaxRunningSteps = 15
    # dirName = os.path.dirname(__file__)
    # evalEnvModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    # evalModel = load_model_from_path(evalEnvModelPath)
    # evalSimulation = MjSim(evalModel)
    # evalKillzoneRadius = 0.5
    # evalIsTerminal = IsTerminal(evalKillzoneRadius, getSheepPos, getWolfPos)
    # evalNumSimulationFrames = 20
    # transit = TransitionFunction(evalSimulation, evalIsTerminal, evalNumSimulationFrames)
    # evalQVelInit = (0, 0, 0, 0)
    # evalNumAgent = 2
    # evalQPosInitNoise = 0
    # evalQVelInitNoise = 0
    # allEvalQPosInit = np.random.uniform(-9.7, 9.7, (evalNumTrials, 4))
    # getResetFromQPosInit = lambda evalQPosInit: Reset(evalSimulation, evalQPosInit, evalQVelInit, evalNumAgent,
    #                                                   evalQPosInitNoise, evalQVelInitNoise)
    # getQPosInitFromTrialIndex = lambda trialIndex: allEvalQPosInit[trialIndex]
    # getResetFromTrialIndex = lambda trialIndex: getResetFromQPosInit(getQPosInitFromTrialIndex(trialIndex))
    # distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    # getSampleTrajectory = lambda trialIndex: SampleTrajectory(evalMaxRunningSteps, transit, evalIsTerminal,
    #                                                           getResetFromTrialIndex(trialIndex), distToAction)

    # # get save path for trajectories
    # sheepPolicyName = 'NN'
    # trajectoryFixedParameters = {'maxRunningSteps': evalMaxRunningSteps, 'numTrials': evalNumTrials,
    #                              'sheepPolicyName': sheepPolicyName}
    # trajectorySaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'NNPolicyVaryHyperParametersSheepEscapeWolfMujoco',
    #                                        'trajectories', 'evaluate')
    # if not os.path.exists(trajectorySaveDirectory):
    #     os.makedirs(trajectorySaveDirectory)
    # trajectoryExtension = '.pickle'
    # getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # # function to generate trajectories for evaluation
    # dummyLearningRate = 0
    # modelToRestoreVariables = generateModel(dummyLearningRate)
    # getModelFromPath = lambda path: restoreVariables(modelToRestoreVariables, path)
    # getModelPathFromParameters = lambda parameters: getModelSavePath(parameters)
    # getModelFromParameters = lambda parameters: getModelFromPath(getModelPathFromParameters(parameters))
    # getPolicyFromParameters = lambda parameters: getPolicy(getModelFromParameters(parameters))

    # generateTrajectories = GenerateTrajectories(evalNumTrials, getPolicyFromParameters, getSampleTrajectory,
    #                                             getTrajectorySavePath, saveData)

    # # # generate trajectories for all conditions
    # toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # # measurement Function
    # initTimeStep = 0
    # stateIndex = 0
    # getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    # optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepPos, getWolfPos,
    #                                                        computeAngleBetweenVectors)
    # getOptimalAction = lambda state: agentDistToGreedyAction(optimalPolicy(state))
    # stationaryAgentAction = lambda state: agentDistToGreedyAction(stationaryAgentPolicy(state))
    # sheepTransit = lambda state, action: transit(state, [action, stationaryAgentAction(state)])
    # computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction, sheepTransit,
    #                                               getSheepPos)
    # measurementTimeStep = 1
    # getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    # getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepPos, getNextStateFromTrajectory)
    # measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos,
    #                                                                   getPosAtNextStepFromTrajectory)

    # # compute statistics on the trajectories
    # loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    # computeStatistics = ComputeStatistics(loadTrajectories, len)
    # statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # print("statisticsDf")
    # print(statisticsDf)

    # plot the results
    fig = plt.figure()
    numColumns = len(manipulatedVariables['miniBatchSize'])
    numRows = len(manipulatedVariables['depth'])
    plotCounter = 1

    for miniBatchSize, grp in statisticsDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')

        for depth, group in grp.groupby('depth'):
            group.index = group.index.droplevel('depth')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            # axForDraw.set_ylim(13, 15)
            axForDraw.set_ylabel('miniBatchSize: {}'.format(miniBatchSize))

            # plt.ylabel('Distance between optimal and actual next position of sheep')
            drawPerformanceLine(group, axForDraw, depth)
            plotCounter += 1

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
