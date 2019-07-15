import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from exec.evaluationFunctions import conditionDfFromParametersDict, GenerateInitQPosUniform, ComputeStatistics
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateValueFunction, \
    ApproximateActionPrior, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle

from collections import OrderedDict
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, policyName):
        wolfPolicy = self.getWolfPolicy(policyName)
        sheepPolicy = self.getSheepPolicy(policyName)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, numSamples, preparePolicy, getSampleTrajectory, saveTrajectories, loadNNModel):
        self.numSamples = numSamples
        self.preparePolicy = preparePolicy
        self.getSampleTrajectory = getSampleTrajectory
        self.saveTrajectories = saveTrajectories
        self.loadNNModel = loadNNModel

    def __call__(self, oneConditionDf):
        policyName = oneConditionDf.index.get_level_values('policyName')[0]
        iteration = oneConditionDf.index.get_level_values('iteration')[0]
        self.loadNNModel(iteration)
        print("Generating trajectories for policy {} and iteration {}".format(policyName, iteration))
        policy = self.preparePolicy(policyName)
        allSampleTrajectories = [self.getSampleTrajectory(sampleIndex) for sampleIndex in range(self.numSamples)]
        allTrajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        self.saveTrajectories(allTrajectories, oneConditionDf)

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['policyName'] = ['bootStrappedMCTSNN', 'heatSeeking', 'NNOnly']
    manipulatedVariables['iteration'] = [1, 200, 400, 600, 800, 999]
    numSimulations = 100
    maxRunningSteps = 20
    numSamples = 30
    killzoneRadius = 2

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

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

    # functions to make predictions from NN
    approximateActionPrior = ApproximateActionPrior(initializedNNModel, actionSpace)
    approximateValueFunction = ApproximateValueFunction(initializedNNModel)
    getStateFromNode = lambda node: list(node.id.values())[0]
    getApproximateValueFromNode = lambda node: approximateValueFunction(getStateFromNode(node))
    terminalReward = 1
    estimateValue = lambda node: terminalReward if isTerminal(getStateFromNode(node)) else getApproximateValueFromNode(node)

    # NN load path
    trainBufferSize = 2000
    trainLearningRate = 0.0001
    trainMaxRunningSteps = 10
    trainNumSimulations = 200
    trainMiniBatchSize = 64
    trainNumTrajectoriesPerIteration = 1
    trainQPosInit = (0, 0, 0, 0)
    trainQPosInitNoise = 9.7
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations,
                         'bufferSize': trainBufferSize, 'learningRate': trainLearningRate,
                         'miniBatchSize': trainMiniBatchSize, 'qPosInit': trainQPosInit,
                         'qPosInitNoise': trainQPosInitNoise, 'numTrajectoriesPerIteration': trainNumTrajectoriesPerIteration}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer', 'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # function to load NN
    loadNNModel = lambda iteration: restoreVariables(initializedNNModel, getNNModelSavePath({'iteration': iteration}))

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    wolfActionInSheepMCTSSimulation = lambda state: (0, 0)
    transitInSheepMCTSSimulation = \
        lambda state, sheepSelfAction: transit(state, [sheepSelfAction, wolfActionInSheepMCTSSimulation(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation, approximateActionPrior)
    expandNNPrior = Expand(isTerminal, initializeChildrenNNPrior)

    # wrapper for sheep policies
    bootStrappedMCTSNN = MCTS(numSimulations, selectChild, expandNNPrior, estimateValue, backup,
                              establishPlainActionDist)
    heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepXPos, getWolfXPos,
                                                               computeAngleBetweenVectors)
    approximatePolicy = ApproximatePolicy(initializedNNModel, actionSpace)
    NNOnly = lambda state: {approximatePolicy(state): 1}
    allSheepPolicies = {'bootStrappedMCTSNN': bootStrappedMCTSNN, 'heatSeeking': heatSeekingPolicy, 'NNOnly': NNOnly}
    getSheepPolicy = lambda policyName: allSheepPolicies[policyName]

    # policy
    getWolfPolicy = lambda policyName: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getWolfPolicy, getSheepPolicy)

    # generate qPosInit
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgents = 2
    getReset = lambda qPosInit, qVelInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents,
                                                  qPosInitNoise, qVelInitNoise)
    getResetFromQPosInit = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgents,
                                                  qPosInitNoise, qVelInitNoise)
    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInit)
    allInitQPos = [generateInitQPos() for _ in range(numSamples)]
    allInitQVel = np.random.uniform(-10, 10, (numSamples, 4))

    # sample trajectory
    getResetFromSample = lambda sample: getReset(allInitQPos[sample], allInitQVel[sample])
    getSampleTrajectory = lambda sample: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                          getResetFromSample(sample), chooseGreedyAction)

    # save path for trajectories
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,
                                 'numSamples': numSamples, 'killzoneRadius': killzoneRadius, 'qVelInitNoise': qVelInitNoise}
    trajectorySaveDirectory = os.path.join(dirName, '..', 'data', 'bootStrappedMCTSNNvsHeatSeekingSheepChaseWolfMujoco',
                                           'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)


    # function to generate trajectories
    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)
    saveAllTrajectoriesFromConditionDf = lambda trajectories, oneConditionDf: \
        saveAllTrajectories(trajectories, readParametersFromDf(oneConditionDf))
    generateTrajectories = GenerateTrajectories(numSamples, preparePolicy, getSampleTrajectory,
                                                saveAllTrajectoriesFromConditionDf, loadNNModel)
    # generate trajectories
    levelNames = list(manipulatedVariables.keys())
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    fig = plt.figure()
    numColumns = 1
    numRows = 1
    pltCounter = 1
    for policy, grp in statisticsDf.groupby('policyName'):
        grp.index = grp.index.droplevel('policyName')
        axForDraw = fig.add_subplot(numRows, numColumns, pltCounter)
        grp.plot(ax=axForDraw, y='mean', marker='o', label=policy)

    plt.title('MCTS+RandomNN vs. Discrete heat seeking policy in chasing task')
    plt.ylabel('Episode Length')
    plt.show()


if __name__ == '__main__':
    main()