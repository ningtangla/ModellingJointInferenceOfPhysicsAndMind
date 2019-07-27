import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from collections import OrderedDict
from matplotlib import pyplot as plt
import mujoco_py as mujoco
import numpy as np

from exec.evaluationFunctions import conditionDfFromParametersDict, ComputeStatistics
from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.algorithms.mcts import RollOut, Expand, InitializeChildren, ScoreChild, SelectChild, MCTS, backup, \
    establishPlainActionDist
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximateValueFunction, \
    ApproximateActionPrior
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    LoadTrajectories, loadFromPickle, readParametersFromDf
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.evaluationFunctions import GenerateInitQPosUniform


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, policyName, numSimulations):
        sheepPolicy = self.getSheepPolicy(policyName, numSimulations)
        wolfPolicy = self.getWolfPolicy(policyName, numSimulations)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, allSampleTrajectories, preparePolicy, saveTrajectories):
        self.allSampleTrajectories = allSampleTrajectories
        self.preparePolicy = preparePolicy
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        policyName = oneConditionDf.index.get_level_values('policy')[0]
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        policy = self.preparePolicy(policyName, numSimulations)
        allTrajectories = [sampleTrajectory(policy) for sampleTrajectory in self.allSampleTrajectories]
        self.saveTrajectories(allTrajectories, oneConditionDf)

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['policy'] = ['MCTSRolloutHeuristic']#['MCTSNN', 'MCTSRolloutHeuristic']
    manipulatedVariables['numSimulations'] = [50, 75]#[50, 100, 200]
    numSamples = 50
    maxRunningSteps = 20

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)
    levelNames = list(manipulatedVariables.keys())

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    sheepActionInMCTS = lambda state: (0, 0)
    transitInWolfMCTS = lambda state, action: transit(state, [sheepActionInMCTS(state), action])

    # Neural Network
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    NNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # restore model
    NNModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco/trainedNNModels/killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    restoreVariables(NNModel, NNModelPath)

    # functions to make predictions from NN
    getStateFromNode = lambda node: list(node.id.values())[0]
    approximateValue = ApproximateValueFunction(NNModel)
    getValueFromNode = lambda node: approximateValue(getStateFromNode(node))
    approximatePrior = ApproximateActionPrior(NNModel, actionSpace)

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTS, getUniformActionPrior)
    expandUniform = Expand(isTerminal, initializeChildrenUniformPrior)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTS, rewardFunction, isTerminal, rolloutHeuristic)

    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInWolfMCTS, approximatePrior)
    expandNNPrior = Expand(isTerminal, initializeChildrenNNPrior)

    # wrappers for wolf policies
    getMCTSNN = lambda numSimulations: MCTS(numSimulations, selectChild, expandNNPrior, getValueFromNode, backup,
                                            establishPlainActionDist)
    getMCTSRolloutHeuristic = lambda numSimulations: MCTS(numSimulations, selectChild, expandUniform, rollout, backup,
                                                          establishPlainActionDist)
    getWolfPolicies = {'MCTSNN': getMCTSNN, 'MCTSRolloutHeuristic': getMCTSRolloutHeuristic}
    getWolfPolicy = lambda policyName, numSimulations: getWolfPolicies[policyName](numSimulations)

    # policy
    getSheepPolicy = lambda policyName, numSimulations: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # sample trajectories
    numAgent = 2
    getResetFromQPosInitDummy = lambda qPosInit: Reset(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
    allQPosInit = [generateQPosInit() for _ in range(numSamples)]
    qVelInitRange = 8
    allQVelInit = np.random.uniform(-qVelInitRange, qVelInitRange, (numSamples, 4))
    getResetFromSampleIndex = lambda sampleIndex: Reset(physicsSimulation, allQPosInit[sampleIndex],
                                                        allQVelInit[sampleIndex], numAgent)
    getSampleTrajectory = lambda sampleIndex: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                               getResetFromSampleIndex(sampleIndex), chooseGreedyAction)
    allSampleTrajectory = [getSampleTrajectory(sampleIndex) for sampleIndex in range(numSamples)]

    # save trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'compareMCTSNNvsMCTSRolloutHeuristicAccumulatedRewardsChaseMujoco', 'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryParameters = {'maxRunningSteps': maxRunningSteps, 'qVelInitNoise': qVelInitRange}
    trajectoryExtension = '.pickle'

    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryParameters)
    generateAllSampleIndexPaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexPaths)
    saveAllTrajectoriesFromDf = lambda trajectories, df: saveAllTrajectories(trajectories, readParametersFromDf(df))

    # generate trajectories
    generateTrajectories = GenerateTrajectories(allSampleTrajectory, preparePolicy, saveAllTrajectoriesFromDf)
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
    axForDraw = fig.add_subplot(1, 1, 1)

    for policyName, grp in statisticsDf.groupby('policy'):
        grp.index = grp.index.droplevel('policy')
        grp.plot(y='mean', marker='o', label=policyName, ax=axForDraw)

    plt.ylabel('accumulated value')
    plt.title('Chasing task')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()