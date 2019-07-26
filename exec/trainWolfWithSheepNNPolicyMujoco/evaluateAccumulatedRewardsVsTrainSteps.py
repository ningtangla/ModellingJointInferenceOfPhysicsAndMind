import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import mujoco_py as mujoco
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import TransitionFunction, Reset, IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from exec.evaluationFunctions import GenerateInitQPosUniform, conditionDfFromParametersDict, ComputeStatistics
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy


class GetWolfPolicy:
    def __init__(self, wolfNNModel, wolfPolicy, allGetWolfModelPaths, restoreVariables):
        self.wolfNNModel = wolfNNModel
        self.wolfPolicy = wolfPolicy
        self.allGetWolfModelPaths = allGetWolfModelPaths
        self.restoreVariables = restoreVariables

    def __call__(self, policyName, trainSteps):
        getWolfModelPath = self.allGetWolfModelPaths[policyName]
        wolfModelPath = getWolfModelPath(trainSteps)
        self.restoreVariables(self.wolfNNModel, wolfModelPath)

        return self.wolfPolicy


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, oneConditionDf):
        trainSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        policyName = oneConditionDf.index.get_level_values('policyName')[0]

        sheepPolicy = self.getSheepPolicy(policyName, trainSteps)
        wolfPolicy = self.getWolfPolicy(policyName, trainSteps)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, allSampleTrajectories, preparePolicy, saveTrajectories):
        self.allSampleTrajectories = allSampleTrajectories
        self.preparePolicy = preparePolicy
        self.saveTrajectories = saveTrajectories

    def __call__(self, oneConditionDf):
        policy = self.preparePolicy(oneConditionDf)
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in self.allSampleTrajectories]
        self.saveTrajectories(trajectories, oneConditionDf)

        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['trainSteps'] = list(range(0, 200000, 25000)) + [200000-1]
    manipulatedVariables['policyName'] = ['trainWithSheepConstantVelocity', 'trainWithSheepEscapePolicy']
    numSamples = 250
    maxRunningSteps = 20

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

    # env mujoco
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numAgent = 2

    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgentsEqualMass.xml')
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

    # NN Model
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # sheep policy
    sheepModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainNNEscapePolicyMujoco/trainedNNModels/killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999'
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepApproximatePolicy = ApproximatePolicy(sheepNNModel, actionSpace)
    sheepPolicy = lambda state: {sheepApproximatePolicy(state): 1}

    # get evaluation wolf NN path
    trainDataNumSimulations = 125
    trainDataKillzoneRadius = 2
    learningRate = 0.0001
    miniBatchSize = 256
    NNModelSaveParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius,
                             'learningRate': learningRate, 'miniBatchSize': miniBatchSize}
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'trainWolfWithSheepNNPolicyMujoco', 'trainedNNModels2000TrainTrajectories')
    NNModelExtension = ''
    getNNModelPath = GetSavePath(NNModelSaveDirectory, NNModelExtension, NNModelSaveParameters)

    # sample trajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    getResetFromInitQPosDummy = lambda qPosInit: Reset(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent,
                                                       qPosInitNoise, qVelInitNoise)
    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromInitQPosDummy)
    allInitQPos = [generateInitQPos() for _ in range(numSamples)]
    qVelInitRange = 8
    allInitQVel = np.random.uniform(-qVelInitRange, qVelInitRange, (numSamples, 4))
    getResetFromTrial = lambda trial: Reset(physicsSimulation, allInitQPos[trial], allInitQVel[trial], numAgent,
                                            qPosInitNoise, qVelInitNoise)
    getSampleTrajectoryFromTrial = lambda trial: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                                  getResetFromTrial(trial), chooseGreedyAction)
    allSampleTrajectories = [getSampleTrajectoryFromTrial(trial) for trial in range(numSamples)]

    # save evaluation trajectories
    numTrainTrajectories = 2000
    trajectorySaveParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius,
                                'learningRate': learningRate, 'miniBatchSize': miniBatchSize,
                                'numTrainTrajectories': numTrainTrajectories}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'trainWolfWithSheepNNPolicyMujoco', 'evaluationTrajectoriesBothAgentsEqualMass')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectorySaveParameters)

    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)
    saveAllTrajectoriesFromDf = lambda trajectories, df: saveAllTrajectories(trajectories, readParametersFromDf(df))

    # policy
    getSheepPolicy = lambda policyName, trainSteps: sheepPolicy

    getBaselineWolfModelSavePath = lambda trainSteps: '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/trainedNNModels/bufferSize=2000_iteration=19999_learningRate=0.0001_maxRunningSteps=20_miniBatchSize=256_numSimulations=200_numTrajectoriesPerIteration=1'
    getEvaluationWolfModelSavePath = lambda trainSteps: getNNModelPath({'trainSteps': trainSteps})
    allGetWolfNNModelSavePaths = {'trainWithSheepConstantVelocity': getBaselineWolfModelSavePath,
                                  'trainWithSheepEscapePolicy': getEvaluationWolfModelSavePath}
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    wolfApproximatePolicy = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = lambda state: {wolfApproximatePolicy(state): 1}
    getWolfPolicy = GetWolfPolicy(wolfNNModel, wolfPolicy, allGetWolfNNModelSavePaths, restoreVariables)

    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # generate trajectories
    generateTrajectories = GenerateTrajectories(allSampleTrajectories, preparePolicy, saveAllTrajectoriesFromDf)
    levelNames = list(manipulatedVariables.keys())
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    decay = 1

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    for policyName, grp in statisticsDf.groupby('policyName'):
        grp.index = grp.index.droplevel('policyName')
        grp.plot(y='mean', marker='o', label=policyName, ax=axis)

    plt.ylabel('Accumulated rewards')
    plt.title('Training wolf with sheep using NN Policy to escape\nBoth agents have equal mass in chasing task')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()