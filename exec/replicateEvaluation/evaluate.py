import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from collections import OrderedDict
import mujoco_py as mujoco
import numpy as np
from matplotlib import pyplot as plt
import glob

from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform, TransitionFunction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.trajectoriesSaveLoad import conditionDfFromParametersDict, GetSavePath, saveToPickle, \
    readParametersFromDf, LoadTrajectories, loadFromPickle
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.preProcessing import AccumulateRewards
from src.neuralNetwork.visualizeNN import syncLimits


class GenerateAllSampleIndexSavePaths:
    def __init__(self, getSavePath):
        self.getSavePath = getSavePath

    def __call__(self, numSamples, pathParameters):
        parametersWithSampleIndex = lambda sampleIndex: dict(
            list(pathParameters.items()) + [('sampleIndex', sampleIndex)])
        genericSavePath = self.getSavePath(parametersWithSampleIndex('*'))
        existingFilesNames = glob.glob(genericSavePath)
        numExistingFiles = len(existingFilesNames)
        allIndexParameters = {sampleIndex: parametersWithSampleIndex(sampleIndex + numExistingFiles) for sampleIndex in
                              range(numSamples)}
        allSavePaths = {sampleIndex: self.getSavePath(indexParameters) for sampleIndex, indexParameters in
                        allIndexParameters.items()}
        return allSavePaths


class SaveAllTrajectories:
    def __init__(self, saveData, generateAllSampleIndexSavePaths):
        self.saveData = saveData
        self.generateAllSampleIndexSavePaths = generateAllSampleIndexSavePaths

    def __call__(self, trajectories, pathParameters):
        numSamples = len(trajectories)
        allSavePaths = self.generateAllSampleIndexSavePaths(numSamples, pathParameters)
        saveTrajectory = lambda sampleIndex: self.saveData(trajectories[sampleIndex], allSavePaths[sampleIndex])
        [saveTrajectory(sampleIndex) for sampleIndex in range(numSamples)]
        return None


def drawPerformanceLine(df, axForDraw):
    trainStepsList = df.index.get_level_values('trainSteps').values
    axForDraw.plot(trainStepsList, [1.15] * len(trainStepsList), label='mcts')
    for nnStruct, nnStructGrp in df.groupby('nnStructure'):
        nnStructGrp.index = nnStructGrp.index.droplevel('nnStructure')
        nnStructGrp.plot(ax=axForDraw, y='mean', marker='o', label=f'NN structure={nnStruct}')


class NeuralNetPolicy:
    def __init__(self, actionSpace, restoreNN):
        self.actionSpace = actionSpace
        self.restoreNN = restoreNN

    def __call__(self, nnStructure, numTrajectories, lossCoefs, trainSteps):
        model = self.restoreNN(nnStructure, numTrajectories, lossCoefs, trainSteps)
        approximateSheepPolicy = ApproximatePolicy(model, self.actionSpace)
        approximateSheepPolicyActionDist = lambda state: {approximateSheepPolicy(state): 1}
        return approximateSheepPolicyActionDist


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, nnStructure, numTrajectories, lossCoefs, trainSteps):
        sheepPolicy = self.getSheepPolicy(nnStructure, numTrajectories, lossCoefs, trainSteps)
        wolfPolicy = self.getWolfPolicy()
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
        return policy


class GenerateTrajectories:
    def __init__(self, preparePolicy, getSampleTrajectories, saveTrajectories):
        self.preparePolicy = preparePolicy
        self.getSampleTrajectories = getSampleTrajectories
        self.saveTrajectories = saveTrajectories
        self.mctsFlag = False

    def __call__(self, oneConditionDf):
        nnStructure = oneConditionDf.index.get_level_values('nnStructure')[0]
        numTrajectories = oneConditionDf.index.get_level_values('numTrainingTrajectories')[0]
        lossCoefs = oneConditionDf.index.get_level_values('lossCoefs')[0]
        trainingSteps = oneConditionDf.index.get_level_values('trainSteps')[0]
        policy = self.preparePolicy(nnStructure, numTrajectories, lossCoefs, trainingSteps)
        allSampleTrajectories = self.getSampleTrajectories()
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        self.saveTrajectories(trajectories, oneConditionDf)
        return None


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['nnStructure'] = [8]
    manipulatedVariables['lossCoefs'] = [(1,1)]
    manipulatedVariables['trainSteps'] = [0,20000,40000,60000,80000,100000]
    manipulatedVariables['numTrainingTrajectories'] = [4500]

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
    killzoneRadius = 2
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    # wolf policy
    numStateSpace = 12
    wolfSharedWidths = [128]
    wolfActionLayerWidths = [128]
    wolfValueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace)
    wolfNNModel = generateModel(wolfSharedWidths, wolfActionLayerWidths, wolfValueLayerWidths)
    wolfNNModelPath = '../../data/compareTrainingDataSizes/wolfNNModels/killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    restoreVariables(wolfNNModel, wolfNNModelPath)
    approximateWolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = lambda state: {approximateWolfPolicy(state): 1}

    # sheep policy
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'compareNNStructures', 'trainedNNs')
    NNModelSaveExtension = ''
    regularizationFactor = 1e-4
    learningRate = 0.0001
    miniBatchSize = 64
    NNFixedParameters = {'agentId': 0, 'learningRate': learningRate,
                         'miniBatchSize': miniBatchSize, 'maxRunningSteps':25, 'numSimulations':100}
    dirName = os.path.dirname(__file__)
    getNNSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    restoreNN = lambda nnStructure, numTrajectories, lossCeofs, trainSteps:\
        restoreVariables(generateModel([128]*nnStructure, [128],[128]),
                         getNNSavePath({'depth': nnStructure,  'trainSteps': trainSteps}))
    # world policy
    getWolfPolicy = lambda: wolfPolicy
    getSheepPolicy = NeuralNetPolicy(actionSpace, restoreNN)
    preparePolicy = PreparePolicy(getSheepPolicy, getWolfPolicy)

    # sample trajectory
    evalNumSamples = 500
    evalMaxRunningSteps = 25

    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    getResetFromInitQPosDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromInitQPosDummy)
    allQPosInit = [generateQPosInit() for _ in range(evalNumSamples)]
    allQVelInit = np.random.uniform(-8, 8, (evalNumSamples, 4))
    getResetFromSampleIndex = lambda sampleIndex: ResetUniform(physicsSimulation, allQPosInit[sampleIndex],
                                                               allQVelInit[sampleIndex], numAgent, qPosInitNoise,
                                                               qVelInitNoise)
    getSampleTrajectory = lambda maxRunningSteps, sampleIndex: SampleTrajectory(maxRunningSteps, transit, isTerminal,
                                                                         getResetFromSampleIndex(sampleIndex),
                                                                         chooseGreedyAction)
    getAllSampleTrajectories = lambda: [getSampleTrajectory(evalMaxRunningSteps, sampleIndex) for
                                        sampleIndex in range(evalNumSamples)]

    # saving trajectories
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'compareNNStructures',
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
    levelNames = list(manipulatedVariables.keys())
    # generateTrajectories = GenerateTrajectories(preparePolicy, getAllSampleTrajectories, saveTrajectoriesFromOneConditionDf)
    # toSplitFrame.groupby(levelNames).apply(generateTrajectories)

    # compute statistics
    fuzzyParameters = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzyParameters)
    loadTrajectoriesFromDf = lambda oneConditionDf: loadTrajectories(readParametersFromDf(oneConditionDf))
    decay = 1
    alivePenalty = 0.05
    deathBonus = -1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    saveToPickle(statisticsDf, 'tempDf.pkl')

    # plot
    fig = plt.figure()
    numColumns = len(manipulatedVariables['lossCoefs'])
    numRows = 1
    plotCounter = 1

    axs = []
    for lossCoefs, lossCoefsGrp in statisticsDf.groupby('lossCoefs'):
        lossCoefsGrp.index = lossCoefsGrp.index.droplevel('lossCoefs')
        for numTrajs, grp in lossCoefsGrp.groupby('numTrainingTrajectories'):
            grp.index = grp.index.droplevel('numTrainingTrajectories')
            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            axForDraw.set_title("maxRunningSteps={}, #trajectories={}, lossCoefs={}, regFactor={}"
                                .format(evalMaxRunningSteps, numTrajs, lossCoefs, regularizationFactor))
            axForDraw.set_ylabel('accumulated rewards')
            drawPerformanceLine(grp, axForDraw)
            axs.append(axForDraw)
            plotCounter += 1
            plt.legend(loc='best')

    syncLimits(axs)
    plt.show()


if __name__ == '__main__':
    main()
