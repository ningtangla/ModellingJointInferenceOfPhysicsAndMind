import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from collections import OrderedDict
import pickle
import pandas as pd
import time
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximateActionPrior, \
    ApproximateValueFunction
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode


def drawPerformanceLine(dataDf, axForDraw, numSimulations, trainSteps):
    for key, grp in dataDf.groupby('modelLearnRate'):
        grp.index = grp.index.droplevel('modelLearnRate')
        grp.plot(ax=axForDraw, title='numSimulations={}, trainSteps={}'.format(numSimulations, trainSteps), y='mean',
                 yerr='std', marker='o', label='learnRate={}'.format(key))


class RestoreNNModel:
    def __init__(self, getModelSavePath, NNModel, restoreVariables):
        self.getModelSavePath = getModelSavePath
        self.NNmodel = NNModel
        self.restoreVariables = restoreVariables

    def __call__(self, iteration):
        modelPath = self.getModelSavePath({'iteration': iteration})
        restoredNNModel = self.restoreVariables(self.NNmodel, modelPath)

        return restoredNNModel


def saveData(data, path):
    pickleOut = open(path, 'wb')
    pickle.dump(data, pickleOut)
    pickleOut.close()


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, NNModel):
        wolfPolicy = self.getWolfPolicy(NNModel)
        sheepPolicy = self.getSheepPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, numTrials, getSampleTrajectory, getSavePath, restoreNNModel, preparePolicy, saveData,
                 readParametersFromDf):
        self.numTrials = numTrials
        self.getSampleTrajectory = getSampleTrajectory
        self.getSavePath = getSavePath
        self.restoreNNModel = restoreNNModel
        self.preparePolicy = preparePolicy
        self.saveData = saveData
        self.readParametersFromDf = readParametersFromDf

    def __call__(self, oneConditionDf):
        startTime = time.time()
        iteration = oneConditionDf.index.get_level_values('iteration')[0]
        restoredNNModel = self.restoreNNModel(iteration)
        policy = self.preparePolicy(restoredNNModel)
        parameters = self.readParametersFromDf(oneConditionDf)
        allSampleTrajectories = {sampleIndex: self.getSampleTrajectory(sampleIndex) for sampleIndex in range(self.numTrials)}
        trajectories = {sampleIndex: sampleTrajectory(policy) for sampleIndex, sampleTrajectory in allSampleTrajectories.items()}
        parametersWithSampleIndex = lambda sampleIndex: dict(list(parameters.items()) + [('sampleIndex', sampleIndex)])
        allIndexParameters = {sampleIndex: parametersWithSampleIndex(sampleIndex) for sampleIndex in range(self.numTrials)}
        allSavePaths = {sampleIndex: self.getSavePath(indexParameters) for sampleIndex, indexParameters in allIndexParameters.items()}
        [self.saveData(trajectories[sampleIndex], allSavePaths[sampleIndex]) for sampleIndex in range(self.numTrials)]

        endTime = time.time()
        print("Time for iteration {} = {}".format(iteration, (endTime-startTime)))

        return None


def main():
    # manipulated variables (and some other parameters that are commonly varied)
    evalNumSimulations = 75#50
    evalNumTrials = 30
    evalMaxRunningSteps = 15#2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['iteration'] = [0, 50, 200, 400, 600, 800, 999]#[0, 50, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                                         #5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 9999]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    numAgents = 2

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 2#0.5
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

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    trainMaxRunningSteps = 15#10
    trainNumSimulations = 150#100
    trainLearningRate = 0.001#0.0001
    trainBufferSize = 2000
    trainMiniBatchSize = 64
    trainNumTrajectoriesPerIteration = 1
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations,
                         'bufferSize': trainBufferSize, 'learningRate': trainLearningRate,
                         'miniBatchSize': trainMiniBatchSize,
                         'numTrajectoriesPerIteration': trainNumTrajectoriesPerIteration}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                        'replayBufferEscape', 'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # functions to get prediction from NN
    getApproximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                                   getApproximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))
    getStateFromNode = lambda node: list(node.id.values())[0]
    terminalReward = -1
    getEstimateValue = lambda NNModel: \
        EstimateValueFromNode(terminalReward, isTerminal, getStateFromNode, ApproximateValueFunction(NNModel))

    # wrapper function for policy
    getMCTSNNPriorValue = lambda NNModel: MCTS(evalNumSimulations, selectChild, getExpandNNPrior(NNModel),
                                               getEstimateValue(NNModel), backup, establishPlainActionDist)
    # getWolfPolicy = lambda NNModel: stationaryAgentPolicy
    wolfActionMagnitude = 5
    heatSeekingWolfPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfXPos, getSheepXPos, wolfActionMagnitude)
    getWolfPolicy = lambda NNModel: heatSeekingWolfPolicy
    preparePolicy = PreparePolicy(getWolfPolicy, getMCTSNNPriorValue)

    # sample trajectory
    evalQPosInitNoise = 0
    evalQVelInitNoise = 0
    # evalQVelInit = (0, 0, 0, 0)
    # evalQVelInitNoise = 1
    getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgents,
                                                              evalQPosInitNoise, 0)

    # generate a set of starting conditions to maintain consistency across all the conditions
    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
    evalAllQPosInit = [generateInitQPos() for _ in range(evalNumTrials)]
    evalAllQVelInit = np.random.uniform(-1, 1, (evalNumTrials, 4))
    getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, evalAllQPosInit[trial], evalAllQVelInit[trial],
                                                   numAgents, evalQPosInitNoise, evalQVelInitNoise)
    getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal,
                                                         getResetFromTrial(trial), chooseGreedyAction)

    # path to save evaluation trajectories
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                       'replayBufferEscape', 'evaluationTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': evalMaxRunningSteps, 'numTrials': evalNumTrials,
                                 'trainNumSimulations': trainNumSimulations, 'trainLearningRate': trainLearningRate,
                                 'trainBufferSize': trainBufferSize, 'trainMiniBatchSize': trainMiniBatchSize,
                                 'trainNumTrajectoriesPerIteration': trainNumTrajectoriesPerIteration}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # function to generate trajectories
    restoreNNModelFromIteration = RestoreNNModel(getNNModelSavePath, initializedNNModel, restoreVariables)
    generateTrajectories = GenerateTrajectories(evalNumTrials, getSampleTrajectory, getTrajectorySavePath,
                                                restoreNNModelFromIteration, preparePolicy, saveData, readParametersFromDf)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # # measurement Function
    # initTimeStep = 0
    # stateIndex = 0
    # getInitStateFromTrajectory = GetStateFromTrajectory(initTimeStep, stateIndex)
    # optimalPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepXPos, getWolfXPos, computeAngleBetweenVectors)
    # getOptimalAction = lambda state: chooseGreedyAction(optimalPolicy(state))
    # computeOptimalNextPos = ComputeOptimalNextPos(getInitStateFromTrajectory, getOptimalAction,
    #                                               transitInSheepMCTSSimulation, getSheepXPos)
    # measurementTimeStep = 1
    # getNextStateFromTrajectory = GetStateFromTrajectory(measurementTimeStep, stateIndex)
    # getPosAtNextStepFromTrajectory = GetAgentPosFromTrajectory(getSheepXPos, getNextStateFromTrajectory)
    # measurementFunction = DistanceBetweenActualAndOptimalNextPosition(computeOptimalNextPos, getPosAtNextStepFromTrajectory)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    print("statisticsDf")
    print(statisticsDf)

    # plot the statistics
    statisticsDf.plot(y='mean', marker='o')
    plt.ylabel("Episode length in escaping task")
    plt.title("Evaluating the performance of MCTS+NN bootstrapping as a function of training iterations")
    plt.show()


if __name__ == '__main__':
    main()