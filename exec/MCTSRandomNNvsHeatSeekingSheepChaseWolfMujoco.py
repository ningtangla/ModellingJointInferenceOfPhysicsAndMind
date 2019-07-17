import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

from exec.evaluationFunctions import conditionDfFromParametersDict, GenerateInitQPosUniform, ComputeStatistics
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximatePolicy, ApproximateValueFunction, \
    ApproximateActionPrior
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
    def __init__(self, numSamples, preparePolicy, getSampleTrajectory, saveTrajectories):
        self.numSamples = numSamples
        self.preparePolicy = preparePolicy
        self.getSampleTrajectory = getSampleTrajectory
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        policyName = oneConditionDf.index.get_level_values('policyName')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        print("Generating trajectories for policy {} and numSim {}".format(policyName, numSimulations))
        policy = self.preparePolicy(policyName, numSimulations)
        allSampleTrajectories = [self.getSampleTrajectory(sampleIndex) for sampleIndex in range(self.numSamples)]
        allTrajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        self.saveTrajectories(allTrajectories, oneConditionDf)

        return None

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['policyName'] = ['MCTSRandomNN', 'heatSeeking']
    manipulatedVariables['numSimulations'] = [1, 100, 200]
    numSamples = 30
    maxRunningSteps = 10

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

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qVelInitNoise = 2

    killzoneRadius = 2
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
    getHeatSeekingPolicy = lambda numSimulations: HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepXPos,
                                                                                         getWolfXPos, computeAngleBetweenVectors)
    allSheepPolicyWrappers = {'MCTSRandomNN': getMCTSRandomNN, 'heatSeeking': getHeatSeekingPolicy}
    getSheepPolicy = lambda policyName, numSimulations: allSheepPolicyWrappers[policyName](numSimulations)

    # policy
    getWolfPolicy = lambda policyName, numSimulations: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getWolfPolicy, getSheepPolicy)

    # generate qPosInit
    qPosInitNoise = 0
    getResetFromQPosInit = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents,
                                                  qPosInitNoise, qVelInitNoise)
    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInit)
    allInitQPos = [generateInitQPos() for _ in range(numSamples)]

    # sample trajectory
    getResetFromSample = lambda sample: getResetFromQPosInit(allInitQPos[sample])
    getSampleTrajectory = lambda sample: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                          getResetFromSample(sample), chooseGreedyAction)

    # save path for trajectories
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps}
    trajectorySaveDirectory = os.path.join(dirName, '..', 'data', 'MCTSRandomNNvsHeatSeekingSheepChaseWolf',
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
                                                saveAllTrajectoriesFromConditionDf)
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