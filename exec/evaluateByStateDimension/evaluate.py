import sys
import os
src = os.path.join(os.pardir, os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))

from episode import SampleTrajectory, chooseGreedyAction
from constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniform, TransitionFunction
from evaluationFunctions import GenerateInitQPosUniform
from pylab import plt
from trajectoriesSaveLoad import GetSavePath
from collections import OrderedDict
from reward import RewardFunctionCompete
from preProcessing import AccumulateRewards
from evaluateByStateDimension.preprocessData import ZeroValueInState
import policyValueNet as net
import mujoco_py as mujoco
import state
import numpy as np
import pandas as pd
import pickle


def dictToFileName(parameters):
    sortedParameters = sorted(parameters.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters
    ]
    modelName = '_'.join(nameValueStringPairs).replace(" ", "")
    return modelName


class ModifyEscaperInputState:

    def __init__(self, removeIndex, numOfFrame, stateDim, zeroValueInState):
        self.removeIndex = removeIndex
        self.numOfFrame = numOfFrame
        self.stateDim = stateDim
        self.zeroValueInState = zeroValueInState
        self.previousFrame = None

    def __call__(self, worldState):
        state = [np.delete(state, self.removeIndex) for state in worldState]
        nnState = np.asarray(state).flatten()
        if self.previousFrame is None:
            currentFrame = np.concatenate([
                self.zeroValueInState(nnState)
                if num + 1 < self.numOfFrame else nnState
                for num in range(self.numOfFrame)
            ])
            self.previousFrame = currentFrame
            return currentFrame
        else:
            toDeleteIndex = [index for index in range(self.stateDim)]
            deleteLastFrame = np.delete(self.previousFrame, toDeleteIndex)
            currentFrame = np.concatenate([deleteLastFrame, nnState])
            self.previousFrame = currentFrame
        return currentFrame


class PreparePolicy:

    def __init__(self, modifyEscaperInputState):
        self.modifyEscaperInputState = modifyEscaperInputState

    def __call__(self, chaserPolicy, escaperPolicy):
        policy = lambda state: [
            escaperPolicy(self.modifyEscaperInputState(state)),
            chaserPolicy(state)
        ]
        return policy


class EvaluateEscaperPerformance:

    def __init__(self, chaserPolicy, allSampleTrajectory, measure,
                 getGenerateEscaperModel, generateEscaperPolicy,
                 getPreparePolicy, getModelSavePath, getModifyEscaperState):
        self.chaserPolicy = chaserPolicy
        self.allSampleTrajectory = allSampleTrajectory
        self.getGenerateEscaperModel = getGenerateEscaperModel
        self.generateEscaperPolicy = generateEscaperPolicy
        self.getPreparePolicy = getPreparePolicy
        self.getModelSavePath = getModelSavePath
        self.measure = measure
        self.getModifyEscaperState = getModifyEscaperState

    def __call__(self, df):
        neuronsPerLayer = df.index.get_level_values('neuronsPerLayer')[0]
        sharedLayers = df.index.get_level_values('sharedLayers')[0]
        actionLayers = df.index.get_level_values('actionLayers')[0]
        valueLayers = df.index.get_level_values('valueLayers')[0]
        numOfFrame = df.index.get_level_values('numOfFrame')[0]
        numOfStateSpace = df.index.get_level_values('numOfStateSpace')[0]
        indexLevelNames = df.index.names
        parameters = {
            levelName: df.index.get_level_values(levelName)[0]
            for levelName in indexLevelNames
        }
        saveModelDir = self.getModelSavePath(parameters)
        modelName = dictToFileName(parameters)
        modelPath = os.path.join(saveModelDir, modelName)
        generateEscaperModel = self.getGenerateEscaperModel(numOfFrame *
                                                            numOfStateSpace)
        escaperModel = generateEscaperModel([neuronsPerLayer] * sharedLayers,
                                            [neuronsPerLayer] * actionLayers,
                                            [neuronsPerLayer] * valueLayers)
        if os.path.exists(saveModelDir):
            net.restoreVariables(escaperModel, modelPath)
        else:
            return pd.Series({"mean": None})
        modifyEscaperState = self.getModifyEscaperState(numOfFrame,
                                                        numOfStateSpace)
        preparePolicy = self.getPreparePolicy(modifyEscaperState)
        escaperPolicy = self.generateEscaperPolicy(escaperModel)
        policy = preparePolicy(self.chaserPolicy, escaperPolicy)
        trajectories = [
            sampleTraj(policy) for sampleTraj in self.allSampleTrajectory
        ]
        reward = np.mean(
            [self.measure(trajectory) for trajectory in trajectories])
        return pd.Series({"mean": reward})


def main():
    dataDir = os.path.join(os.pardir, os.pardir, 'data',
                           'evaluateByStateDimension')

    # generate policy
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7),
                   (0, -10), (7, -7)]
    chaserSavedModelDir = os.path.join(dataDir, 'wolfNNModels')
    chaserModelName = 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    chaserNumStateSpace = 12
    numActionSpace = 8
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = net.GenerateModel(chaserNumStateSpace, numActionSpace,
                                      regularizationFactor)
    chaserModel = generateModel(sharedWidths, actionLayerWidths,
                                valueLayerWidths)
    net.restoreVariables(chaserModel,
                         os.path.join(chaserSavedModelDir, chaserModelName))
    approximateWolfPolicy = net.ApproximatePolicy(chaserModel, actionSpace)
    chaserPolicy = lambda state: approximateWolfPolicy(state)

    # mujoco
    physicsDynamicsPath = os.path.join(os.pardir, os.pardir, 'env', 'xmls',
                                       'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = state.GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = state.GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal,
                                 numSimulationFrames)

    # sampleTrajectory
    qPosInitNoise = 0
    qVelInitNoise = 0
    numAgent = 2
    getResetFromInitQPosDummy = lambda qPosInit: ResetUniform(
        physicsSimulation, qPosInit, (0, 0, 0, 0), numAgent)
    generateQPosInit = GenerateInitQPosUniform(-9.7, 9.7, isTerminal,
                                               getResetFromInitQPosDummy)
    numTrials = 100
    allQPosInit = [generateQPosInit() for _ in range(numTrials)]
    allQVelInit = np.random.uniform(-8, 8, (numTrials, 4))
    getResetFromSampleIndex = lambda sampleIndex: ResetUniform(
        physicsSimulation, allQPosInit[sampleIndex], allQVelInit[sampleIndex],
        numAgent, qPosInitNoise, qVelInitNoise)
    maxRunningSteps = 25
    getSampleTrajectory = lambda sampleIndex: SampleTrajectory(
        maxRunningSteps, transit, isTerminal,
        getResetFromSampleIndex(sampleIndex), chooseGreedyAction)
    allSampleTrajectory = [
        getSampleTrajectory(sampleIndex) for sampleIndex in range(numTrials)
    ]

    # statistic reward function
    alivePenalty = 0.05
    deathBonus = -1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)
    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measure = lambda trajectory: accumulateRewards(trajectory)[0]

    # split
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionLabel']
    independentVariables['trajectory'] = [4500]
    independentVariables['batchSize'] = [64]
    independentVariables['augment'] = [False]
    independentVariables['trainingStep'] = [
        num for num in range(0, 500001, 50000)
    ]
    independentVariables['neuronsPerLayer'] = [128]
    independentVariables['sharedLayers'] = [1, 2, 4, 8]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]
    independentVariables['numOfFrame'] = [1, 3]
    independentVariables['numOfStateSpace'] = [8]
    independentVariables['lr'] = [1e-3, 1e-4, 1e-5]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    trainedModelDir = os.path.join(dataDir, "trainedModel")
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)
    getModelSavePath = GetSavePath(trainedModelDir, "")
    getGenerateEscaperModel = lambda numStateSpace: net.GenerateModel(
        numStateSpace, numActionSpace, regularizationFactor)
    generateEscaperPolicy = lambda model: net.ApproximatePolicy(
        model, actionSpace)
    qPosIndex = [0, 1]
    zeroIndex = [2, 3, 6, 7]
    zeroValueInState = ZeroValueInState(zeroIndex)
    getModifyEscaperState = lambda numOfFrame, stateDim: ModifyEscaperInputState(
        qPosIndex, numOfFrame, stateDim, zeroValueInState)
    evaluate = EvaluateEscaperPerformance(chaserPolicy, allSampleTrajectory,
                                          measure, getGenerateEscaperModel,
                                          generateEscaperPolicy, PreparePolicy,
                                          getModelSavePath,
                                          getModifyEscaperState)
    statDF = toSplitFrame.groupby(levelNames).apply(evaluate)
    with open(os.path.join(dataDir, "evaluate.pkl"), 'wb') as f:
        pickle.dump(statDF, f)

    # plotbatchSize
    xStatistic = "trainingStep"
    yStatistic = "mean"
    lineStatistic = "numOfFrame"
    rowSubplotStatistic = "lr"
    colSubplotStatistic = "sharedLayers"
    figsize = (12, 10)
    figure = plt.figure(figsize=figsize)
    numOfPlot = 1
    ylimTop = max(statDF[yStatistic]) + 0.2
    ylimBot = min(statDF[yStatistic]) - 0.2
    rowNum = len(statDF.groupby(rowSubplotStatistic))
    colNum = len(statDF.groupby(colSubplotStatistic))
    for rowKey, rowDF in statDF.groupby(rowSubplotStatistic):
        for subplotKey, subplotDF in rowDF.groupby(colSubplotStatistic):
            for linekey, lineDF in subplotDF.groupby(lineStatistic):
                ax = figure.add_subplot(rowNum, colNum, numOfPlot)
                plotDF = lineDF.reset_index()
                subtitle = "sharedLayers:{} LR:{}".format(subplotKey, rowKey)
                plotDF.plot(
                    x=xStatistic,
                    y=yStatistic,
                    ax=ax,
                    label=linekey,
                    title=subtitle)
                plt.ylim(bottom=ylimBot, top=ylimTop)
                plt.ylabel("rewards")
            numOfPlot += 1
    plt.legend(loc='best')
    plt.subplots_adjust(hspace=0.4, wspace=0.6)
    plt.suptitle("batchSize:64, trajectory:6000")
    figureName = "effect_inputFrame_on_NNPerformance.png"
    figurePath = os.path.join(dataDir, figureName)
    plt.savefig(figurePath)


if __name__ == '__main__':
    main()
