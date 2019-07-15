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
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, Reset, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, RollOut, backup, \
    establishPlainActionDist
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.preProcessing import AccumulateRewards
from exec.evaluationFunctions import GenerateInitQPosUniform


def drawPerformanceLine(df, axForDraw):
    for policy, grp in df.groupby('policy'):
        grp.index = grp.index.droplevel('policy')
        grp.plot(ax=axForDraw, y='mean', marker='o', label=policy)


class NeuralNetPolicy:
    def __init__(self, approximatePolicy, restoreNNModel):
        self.approximatePolicy = approximatePolicy
        self.restoreNNModel = restoreNNModel

    def __call__(self, trainSteps):
        self.restoreNNModel(trainSteps)
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
        self.mctsFlag = False

    def __call__(self, oneConditionDf):
        policyName = oneConditionDf.index.get_level_values('policy')[0]
        trainingSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        evaluationMaxRunningSteps = oneConditionDf.index.get_level_values('evaluationMaxRunningSteps')[0]
        policy = self.preparePolicy(policyName, trainingSteps)
        if policyName == 'mcts' and self.mctsFlag == False:
            self.mctsFlag = True
            allSampleTrajectories = self.getSampleTrajectories(evaluationMaxRunningSteps)
            trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
            self.saveTrajectories(trajectories, oneConditionDf)
            self.mctsTrajectories = trajectories
        elif policyName == 'mcts' and self.mctsFlag == True:
            self.saveTrajectories(self.mctsTrajectories, oneConditionDf)
        elif policyName != 'mcts':
            allSampleTrajectories = self.getSampleTrajectories(evaluationMaxRunningSteps)
            trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
            self.saveTrajectories(trajectories, oneConditionDf)

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['policy'] = ['mcts', 'NN']
    manipulatedVariables['evaluationMaxRunningSteps'] = [25]
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

    # neural network model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # wolf NN chase policy
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco/trainedNNModels/killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
    approximateWolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}

    # transit in sheep MCTS simulation
    transitInSheepMCTSSimulation = lambda state, action: transit(state, [action, approximateWolfPolicy(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    alivePenalty = 0.05
    deathBonus = -1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = -0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                      rolloutHeuristic)

    numSimulations = 50
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # NNModel for sheep policy
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # NN save path
    trainDataNumSimulations = 100
    trainDataKillzoneRadius = 2
    trainDataQPosInitNoise = 9.7
    trainDataQVelInitNoise = 8
    trainDataRolloutHeuristicWeight = -0.1
    trainDataMaxRunningSteps = 25
    NNFixedParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius,
                         'qPosInitNoise': trainDataQPosInitNoise, 'qVelInitNoise': trainDataQVelInitNoise,
                         'rolloutHeuristicWeight': trainDataRolloutHeuristicWeight, 'maxRunningSteps': trainDataMaxRunningSteps}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainNNEscapePolicyMujoco', 'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # loading neural network
    restoreNNModel = lambda trainSteps: restoreVariables(sheepNNModel, getNNModelSavePath({'trainSteps': trainSteps}))

    # functions to make prediction from NN
    approximateSheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace)
    approximateSheepPolicyActionDist = lambda state: {approximateSheepPolicy(state): 1}

    getMCTS = lambda trainSteps: mcts
    getNNPolicy = NeuralNetPolicy(approximateSheepPolicyActionDist, restoreNNModel)
    allGetSheepPolicy = {'mcts': getMCTS, 'NN': getNNPolicy}
    getWolfPolicy = lambda policyName, trainSteps: wolfPolicy
    getSheepPolicy = lambda policyName, trainSteps: allGetSheepPolicy[policyName](trainSteps)

    # policy
    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # sample trajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    getResetFromInitQPosDummy = lambda qPosInit: Reset(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromInitQPosDummy)
    allQPosInit = [generateQPosInit() for _ in range(evalNumSamples)]
    allQVelInit = np.random.uniform(-8, 8, (evalNumSamples, 4))
    getResetFromSampleIndex = lambda sampleIndex: Reset(physicsSimulation, allQPosInit[sampleIndex],
                                                               allQVelInit[sampleIndex], numAgent, qPosInitNoise,
                                                               qVelInitNoise)
    getSampleTrajectory = lambda maxRunningSteps, sampleIndex: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                                         getResetFromSampleIndex(sampleIndex),
                                                                         chooseGreedyAction)
    getAllSampleTrajectories = lambda maxRunningSteps: [getSampleTrajectory(maxRunningSteps, sampleIndex) for
                                                        sampleIndex in range(evalNumSamples)]

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainNNEscapePolicyMujoco',
                                           'evaluationTrajectories')
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
        plotCounter += 1

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()

