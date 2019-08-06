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
    establishPlainActionDist, RollOut
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectoriesFromIndividualFiles, readParametersFromDf, loadFromPickle, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximateValue, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards


def drawPerformanceLine(dataDf, axForDraw, numSimulations, trainSteps):
    for key, grp in dataDf.groupby('modelLearnRate'):
        grp.index = grp.index.droplevel('modelLearnRate')
        grp.plot(ax=axForDraw, title='numSimulations={}, trainSteps={}'.format(numSimulations, trainSteps), y='mean',
                 yerr='std', marker='o', label='learnRate={}'.format(key))


class RestoreNNModel:
    def __init__(self, allGetModelSavePaths, allNNModels, restoreVariables):
        self.allGetModelSavePaths = allGetModelSavePaths
        self.allNNModels = allNNModels
        self.restoreVariables = restoreVariables

    def __call__(self, iteration, policyName):
        NNModel = self.allNNModels[policyName]
        getModelSavePath = self.allGetModelSavePaths[policyName]
        modelPath = getModelSavePath({'iteration': iteration})
        restoredNNModel = self.restoreVariables(NNModel, modelPath)

        return restoredNNModel


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, policyName):
        sheepPolicy = self.getSheepPolicy(policyName)
        wolfPolicy = self.getWolfPolicy(policyName)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class GenerateTrajectories:
    def __init__(self, allSampleTrajectories, saveTrajectories, restoreNNModel, preparePolicy):
        self.allSampleTrajectories = allSampleTrajectories
        self.saveTrajectories = saveTrajectories
        self.restoreNNModel = restoreNNModel
        self.preparePolicy = preparePolicy
        self.mctsFlag = False

    def __call__(self, oneConditionDf):
        startTime = time.time()
        iteration = oneConditionDf.index.get_level_values('iteration')[0]
        policyName = oneConditionDf.index.get_level_values('policyName')[0]
        self.restoreNNModel(iteration, policyName)
        policy = self.preparePolicy(policyName)
        if policyName == 'mctsHeuristic' and self.mctsFlag == False:
            trajectories = [sampleTrajectory(policy) for sampleTrajectory in self.allSampleTrajectories]
            self.mctsFlag = True
            self.mctsTrajectories = trajectories
        elif policyName == 'mctsHeuristic' and self.mctsFlag == True:
            trajectories = self.mctsTrajectories
        else:
            trajectories = [sampleTrajectory(policy) for sampleTrajectory in self.allSampleTrajectories]
        self.saveTrajectories(trajectories, oneConditionDf)
        endTime = time.time()
        print("Time for iteration {} = {}".format(iteration, (endTime-startTime)))

        return None


def main():
    # manipulated variables (and some other parameters that are commonly varied)
    evalNumSimulations = 200
    evalNumTrials = 500
    evalMaxRunningSteps = 35
    manipulatedVariables = OrderedDict()
    manipulatedVariables['iteration'] = list(range(0, 10000, 500))
    manipulatedVariables['policyName'] = ['NNPolicy4HiddenLayers', 'NNPolicy2HiddenLayers']#['NNPolicy', 'mctsHeuristic']

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgentsTwoObstacles.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    numAgents = 2

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    # sheepActionInSheepMCTSSimulation = lambda state: (0, 0)
    # transitInWolfMCTSSimulation = lambda state, action: transit(state, [sheepActionInSheepMCTSSimulation(state), action])

    # MCTS
    # cInit = 1
    # cBase = 100
    # calculateScore = ScoreChild(cInit, cBase)
    # selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    # getActionPriorUniform = lambda state: {action: 1/numActionSpace for action in actionSpace}
    # initializeChildren = InitializeChildren(actionSpace, transitInWolfMCTSSimulation, getActionPriorUniform)
    # expand = Expand(isTerminal, initializeChildren)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    # rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    # rolloutHeuristicWeight = 0.1
    # maxRolloutSteps = 10
    # rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    # rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
    #                   rolloutHeuristic)

    # mcts = MCTS(evalNumSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel4Layers = generateModel([128, 128, 128], actionLayerWidths, valueLayerWidths)
    initializedNNModel2Layers = generateModel([128], actionLayerWidths, valueLayerWidths)

    trainMaxRunningSteps = 30
    trainNumSimulations = 200
    trainLearningRate = 0.001
    trainBufferSize = 2000
    trainMiniBatchSize = 256
    trainNumTrajectoriesPerIteration = 1
    numTrainStepsPerIteration = 1
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations,
                         'bufferSize': trainBufferSize, 'learningRate': trainLearningRate,
                         'miniBatchSize': trainMiniBatchSize,
                         'numTrajectoriesPerIteration': trainNumTrajectoriesPerIteration, 'numTrainStepsPerIteration': numTrainStepsPerIteration}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory4Layers = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                        'replayBufferStartWithRandomModelChasingObstacle', '4HiddenLayers', 'trainedNNModels')
    NNModelSaveDirectory2Layers = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                        'replayBufferStartWithRandomModelChasingObstacle', '2HiddenLayers', 'trainedNNModels')
    NNModelSaveExtension = ''
    getNNModelSavePath4Layers = GetSavePath(NNModelSaveDirectory4Layers, NNModelSaveExtension, NNFixedParameters)
    getNNModelSavePath2Layers = GetSavePath(NNModelSaveDirectory2Layers, NNModelSaveExtension, NNFixedParameters)

    # functions to get prediction from NN
    NNPolicy4Layers = ApproximatePolicy(initializedNNModel4Layers, actionSpace)
    NNPolicy2Layers = ApproximatePolicy(initializedNNModel2Layers, actionSpace)

    # # wolf policy
    # wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    # wolfNNModelPath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/trainMCTSNNIteratively/replayBufferStartWithTrainedModel/trainedNNModels/bufferSize=2000_iteration=19999_learningRate=0.0001_maxRunningSteps=20_miniBatchSize=256_numSimulations=200_numTrajectoriesPerIteration=1'
    # restoreVariables(wolfNNModel, wolfNNModelPath)
    # wolfActionForState = ApproximatePolicy(wolfNNModel, actionSpace)
    # wolfPolicy = lambda state: {wolfActionForState(state): 1}

    # policy
    allGetWolfPolicies = {'NNPolicy4HiddenLayers': NNPolicy4Layers, 'NNPolicy2HiddenLayers': NNPolicy2Layers}#, 'mctsHeuristic': mcts}
    getWolfPolicy = lambda policyName: allGetWolfPolicies[policyName]
    getSheepPolicy = lambda policyName: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # generate a set of starting conditions to maintain consistency across all the conditions
    evalQPosInitNoise = 0
    evalQVelInitNoise = 0
    getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgents,
                                                       evalQPosInitNoise, 0)
    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
    evalAllQPosInit = [generateInitQPos() for _ in range(evalNumTrials)]
    evalAllQVelInit = np.random.uniform(-8, 8, (evalNumTrials, 4))
    getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, evalAllQPosInit[trial], evalAllQVelInit[trial],
                                            numAgents, evalQPosInitNoise, evalQVelInitNoise)
    getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal,
                                                         getResetFromTrial(trial), chooseGreedyAction)
    allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

    # save evaluation trajectories
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                       'replayBufferStartWithRandomModelChasingObstacle',
                                       'evaluationTrajectories9500TrainSteps')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': evalMaxRunningSteps, 'numTrials': evalNumTrials,
                                 'trainNumSimulations': trainNumSimulations, 'trainLearningRate': trainLearningRate,
                                 'trainBufferSize': trainBufferSize, 'trainMiniBatchSize': trainMiniBatchSize,
                                 'trainNumTrajectoriesPerIteration': trainNumTrajectoriesPerIteration}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getTrajectorySavePath)
    saveAllTrajectories = SaveAllTrajectories(saveToPickle, generateAllSampleIndexSavePaths)
    saveAllTrajectoriesFromDf = lambda trajectories, df: saveAllTrajectories(trajectories, readParametersFromDf(df))

    # function to restore NN Model
    allNNModels = {'NNPolicy4HiddenLayers': initializedNNModel4Layers, 'NNPolicy2HiddenLayers': initializedNNModel2Layers}
    allGetModelSavePaths = {'NNPolicy4HiddenLayers': getNNModelSavePath4Layers, 'NNPolicy2HiddenLayers': getNNModelSavePath2Layers}
    restoreNNModelFromIteration = RestoreNNModel(allGetModelSavePaths, allNNModels, restoreVariables)
    generateTrajectories = GenerateTrajectories(allSampleTrajectories, saveAllTrajectoriesFromDf, restoreNNModelFromIteration,
                                                preparePolicy)

    # run all trials and save trajectories
    toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics on the trajectories
    loadTrajectories = LoadTrajectoriesFromIndividualFiles(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    # plot the statistics
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    for policyName, grp in statisticsDf.groupby('policyName'):
        grp.index = grp.index.droplevel('policyName')
        grp.plot(y='mean', marker='o', label=policyName, ax=axis)

    plt.ylabel('Accumulated rewards')
    plt.title('iterative training in chase task with obstacle. {} train steps per iteration\nStart with random model'.format(numTrainStepsPerIteration))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()