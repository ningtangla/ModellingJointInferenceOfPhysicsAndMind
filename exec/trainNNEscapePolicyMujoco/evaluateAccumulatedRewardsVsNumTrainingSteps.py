import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from collections import OrderedDict
import mujoco_py as mujoco
import numpy as np
from matplotlib import pyplot as plt

from exec.evaluationFunctions import conditionDfFromParametersDict, ComputeStatistics
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, RollOut, backup, \
    establishPlainActionDist
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors


def drawPerformanceLine(df, axForDraw):
    for policy, grp in df.groupby('policy'):
        grp.index = grp.index.droplevel('policy')
        grp.plot(ax=axForDraw, y='mean', marker='o', label=policy)


class NeuralNetPolicy:
    def __init__(self, maxRunningSteps, approximatePolicy, restoreNNModel):
        self.maxRunningSteps = maxRunningSteps
        self.approximatePolicy = approximatePolicy
        self.restoreNNModel = restoreNNModel

    def __call__(self, iteration):
        self.restoreNNModel(self.maxRunningSteps, iteration)
        return self.approximatePolicy


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, policyName, iteration):
        sheepPolicy = self.getSheepPolicy(policyName, iteration)
        wolfPolicy = self.getWolfPolicy(policyName, iteration)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, preparePolicy, getSampleTrajectories, saveTrajectories):
        self.preparePolicy = preparePolicy
        self.getSampleTrajectories = getSampleTrajectories
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        policyName = oneConditionDf.index.get_level_values('policy')[0]
        trainingSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        evaluationMaxRunningSteps = oneConditionDf.index.get_level_values('evaluationMaxRunningSteps')[0]
        policy = self.preparePolicy(policyName, trainingSteps)
        allSampleTrajectories = self.getSampleTrajectories(evaluationMaxRunningSteps)
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        self.saveTrajectories(trajectories, oneConditionDf)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['evaluationMaxRunningSteps'] = [25, 100]
    manipulatedVariables['trainSteps'] = list(range(0, 100000, 10000)) + [100000-1]
    killzoneRadius = 2
    evalNumSamples = 100

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # Mujoco Environment
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
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
    sheepActionInWolfSimulation = lambda state: (0, 0)
    transitInWolfMCTSSimulation = \
        lambda state, wolfSelfAction: transit(state, [sheepActionInWolfSimulation(state), wolfSelfAction])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTSSimulation,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
                      rolloutHeuristic)

    numSimulations = 50
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # neural network model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # NN save path
    trainDataNumSimulations = 100
    trainDataKillzoneRadius = 0.5
    trainDataQPosInitNoise = 9.7
    trainDataQVelInitNoise = 5
    trainDataRolloutHeuristicWeight = 0.1
    NNFixedParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius,
                         'qPosInitNoise': trainDataQPosInitNoise, 'qVelInitNoise': trainDataQVelInitNoise,
                         'rolloutHeuristicWeight': trainDataRolloutHeuristicWeight}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco',
                                        'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # loading neural network
    restoreNNModel = lambda maxRunningSteps, iteration: \
        restoreVariables(initializedNNModel, getNNModelSavePath({'trainSteps': iteration, 'maxRunningSteps': maxRunningSteps}))

    # functions to make prediction from NN
    approximatePolicy = ApproximatePolicy(initializedNNModel, actionSpace)
    approximatePolicyActionDist = lambda state: {approximatePolicy(state): 1}

    # heat seeking policy
    heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(actionSpace, getWolfXPos, getSheepXPos,
                                                               computeAngleBetweenVectors)

    # get wolf policies
    getNeuralNetPolicy = lambda maxRunningSteps: NeuralNetPolicy(maxRunningSteps, approximatePolicyActionDist,
                                                                 restoreNNModel)
    getMCTS = lambda iteration: mcts
    getHeatSeekingPolicy = lambda iteration: heatSeekingPolicy
    allGetWolfPolicy = {'heatSeeking': getHeatSeekingPolicy, 'NNMaxRunningSteps=10': getNeuralNetPolicy(10),
                       'NNMaxRunningSteps=100': getNeuralNetPolicy(100)}
    getWolfPolicy = lambda policyName, iteration: allGetWolfPolicy[policyName](iteration)
    getSheepPolicy = lambda policyName, iteration: stationaryAgentPolicy

    # policy
    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # sample trajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    allQPosInit = np.random.uniform(-9.7, 9.7, (evalNumSamples, 4))
    allQVelInit = np.random.uniform(-5, 5, (evalNumSamples, 4))
    getResetFromSampleIndex = lambda sampleIndex: ResetUniform(physicsSimulation, allQPosInit[sampleIndex],
                                                               allQVelInit[sampleIndex], numAgent, qPosInitNoise,
                                                               qVelInitNoise)
    getSampleTrajectory = lambda maxRunningSteps, sampleIndex: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                                         getResetFromSampleIndex(sampleIndex),
                                                                         chooseGreedyAction)
    getAllSampleTrajectories = lambda maxRunningSteps: [getSampleTrajectory(maxRunningSteps, sampleIndex) for
                                                        sampleIndex in range(evalNumSamples)]

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco',
                                           'evaluateAccumulatedRewardsEvaluationTrajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveParameters = {}
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectorySaveParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)
    saveTrajectoriesFromOneConditionDf = lambda trajectories, oneConditionDf: \
        saveAllTrajectories(trajectories, readParametersFromDf(oneConditionDf))

    # generate trajectories
    generateTrajectories = GenerateTrajectories(preparePolicy, getAllSampleTrajectories, saveTrajectoriesFromOneConditionDf)
    levelNames = list(manipulatedVariables.keys())
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda oneConditionDf: loadTrajectories(readParametersFromDf(oneConditionDf))
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)

    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot
    fig = plt.figure()
    numColumns = len(manipulatedVariables['evaluationMaxRunningSteps'])
    numRows = 1
    plotCounter = 1

    for maxEpisodeLen, grp in statisticsDf.groupby('evaluationMaxRunningSteps'):
        grp.index = grp.index.droplevel('evaluationMaxRunningSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_title("max running steps in evaluation = {}".format(maxEpisodeLen))
        axForDraw.set_ylabel('accumulated rewards')
        drawPerformanceLine(grp, axForDraw)
        axForDraw.set_ylim(-5, 0.5)
        plotCounter += 1

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()

