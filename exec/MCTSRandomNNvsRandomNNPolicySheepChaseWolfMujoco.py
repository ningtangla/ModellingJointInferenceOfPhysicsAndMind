import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateValueFunction, ApproximateActionPrior, \
    ApproximatePolicy
from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.evaluationFunctions import ComputeStatistics
from exec.trajectoriesSaveLoad import GenerateAllSampleIndexSavePaths, SaveAllTrajectories, GetSavePath, \
    LoadTrajectories, loadFromPickle, saveToPickle, readParametersFromDf

from collections import OrderedDict
import pandas as pd
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np


def drawPerformanceLine(df, axForDraw):
    for policy, grp in df.groupby('sheepPolicyName'):
        grp.index = grp.index.droplevel('sheepPolicyName')
        grp.plot(y='mean', marker='o', label=policy, ax=axForDraw)

    return None


class GenerateInitQPosGaussian:
    def __init__(self, minQPos, maxQPos, isTerminal, getResetFromInitQPos):
        self.minQPos = minQPos
        self.maxQPos = maxQPos
        self.isTerminal = isTerminal
        self.getResetFromInitQPos = getResetFromInitQPos
        self.numQPosEachAgent = 2

    def __call__(self, stDev):
        while True:
            sheepQPosInit = np.random.uniform(self.minQPos, self.maxQPos, self.numQPosEachAgent)
            wolfQPosInit = np.random.normal(sheepQPosInit, stDev)
            qPosInit = np.concatenate((sheepQPosInit, wolfQPosInit))
            reset = self.getResetFromInitQPos(qPosInit)
            initState = reset()
            if not self.isTerminal(initState):
                break

        return qPosInit


class PreparePolicy:
    def __init__(self, getWolfPolicy, getSheepPolicy):
        self.getWolfPolicy = getWolfPolicy
        self.getSheepPolicy = getSheepPolicy

    def __call__(self, policyName, numSimulations):
        wolfPolicy = self.getWolfPolicy(policyName, numSimulations)
        sheepPolicy = self.getSheepPolicy(policyName, numSimulations)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, numSamples, getSampleTrajectory, preparePolicy, saveAllTrajectories):
        self.numSamples = numSamples
        self.getSampleTrajectory = getSampleTrajectory
        self.preparePolicy = preparePolicy
        self.saveAllTrajectories = saveAllTrajectories

    def __call__(self, oneConditionDf):
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        sheepPolicyName = oneConditionDf.index.get_level_values('sheepPolicyName')[0]
        stDev = oneConditionDf.index.get_level_values('stDev')[0]
        print("generating trajectories for numSim {}, policy {}, std {}".format(numSimulations, sheepPolicyName, stDev))
        policy = self.preparePolicy(sheepPolicyName, numSimulations)
        allSampleTrajectories = {sampleIndex: self.getSampleTrajectory(stDev, sampleIndex) for sampleIndex in range(self.numSamples)}
        trajectories = {sampleIndex: sampleTrajectory(policy) for sampleIndex, sampleTrajectory in allSampleTrajectories.items()}
        self.saveAllTrajectories(trajectories, oneConditionDf)

        return None


def main():
    # manipulated parameters and other important parameters
    numSamples = 30
    maxRunningSteps = 10
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [1, 100, 200]
    manipulatedVariables['sheepPolicyName'] = ['RandomNN', 'MCTSRandomNN']
    manipulatedVariables['stDev'] = [2, 6, 16]

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
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # functions to make predictions from NN
    approximateActionPrior = ApproximateActionPrior(initializedNNModel, actionSpace)
    approximateValueFunction = ApproximateValueFunction(initializedNNModel)
    getStateFromNode = lambda node: list(node.id.values())[0]
    getApproximateValueFromNode = lambda node: approximateValueFunction(getStateFromNode(node))
    terminalReward = 1
    estimateValue = lambda node: terminalReward if isTerminal(getStateFromNode(node)) else getApproximateValueFromNode(node)
    approximatePolicy = ApproximatePolicy(initializedNNModel, actionSpace)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 0

    killzoneRadius = 0.5
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
    getMCTSRandomNN = lambda numSimulations: MCTS(numSimulations, selectChild, expandNNPrior,
                                                  estimateValue, backup, establishPlainActionDist)
    getRandomNNPolicy = lambda numSimulations: lambda state: {approximatePolicy(state): 1}
    allSheepPolicyWrappers = {'MCTSRandomNN': getMCTSRandomNN, 'RandomNN': getRandomNNPolicy}
    getSheepPolicy = lambda policyName, numSimulations: allSheepPolicyWrappers[policyName](numSimulations)

    # policy
    getWolfPolicy = lambda policyName, numSimulations: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getWolfPolicy, getSheepPolicy)

    # generate qPosInit
    qPosInitNoise = 0
    getResetFromQPosInit = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents,
                                                  qPosInitNoise, qVelInitNoise)
    generateInitQPos = GenerateInitQPosGaussian(-9.7, 9.7, isTerminal, getResetFromQPosInit)
    getAllInitQPosFromStDev = lambda std: [generateInitQPos(std) for _ in range(numSamples)]
    allQPosInit = {std: getAllInitQPosFromStDev(std) for std in manipulatedVariables['stDev']}

    # sample trajectory
    getResetFromSample = lambda std, sample: getResetFromQPosInit(allQPosInit[std][sample])
    getSampleTrajectory = lambda std, sample: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                          getResetFromSample(std, sample), chooseGreedyAction)

    # save path for trajectories
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps}
    trajectorySaveDirectory = os.path.join(dirName, '..', 'data', 'MCTSRandomNNvsRandomNNPolicySheepChaseWolf',
                                           'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # generate trajectories
    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)
    saveAllTrajectoriesFromConditionDf = lambda trajectories, oneConditionDf: \
        saveAllTrajectories(trajectories, readParametersFromDf(oneConditionDf))
    generateTrajectories = GenerateTrajectories(numSamples, getSampleTrajectory, preparePolicy,
                                                saveAllTrajectoriesFromConditionDf)
    # toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, len)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    fig = plt.figure()
    numColumns = len(manipulatedVariables['stDev'])
    numRows = 1
    pltCounter = 1
    for stDev, grp in statisticsDf.groupby('stDev'):
        grp.index = grp.index.droplevel('stDev')
        axForDraw = fig.add_subplot(numRows, numColumns, pltCounter)
        drawPerformanceLine(grp, axForDraw)
        axForDraw.set_title('Std. Dev. = {}'.format(stDev))
        axForDraw.set_ylabel('Episode Length')
        axForDraw.set_ylim(5, 10.5)
        pltCounter += 1

    plt.show()


if __name__ == '__main__':
    main()
