import sys
import os
src = os.path.join(os.pardir, 'src')
sys.path.append(src)
sys.path.append(os.path.join(src, 'neuralNetwork'))
sys.path.append(os.path.join(src, 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join(src, 'algorithms'))
sys.path.append("../src")
import policyValueNet as net
import envNoPhysics as env
import state
import numpy as np
import pandas as pd
from measure import calculateCrossEntropy
from analyticGeometryFunctions import transitePolarToCartesian, computeAngleBetweenVectors
from pylab import plt
import policies
import mcts
import os
import math
from collections import OrderedDict
from evaluationFunctions import GetSavePath, ComputeStatistics, LoadMultipleTrajectoriesFile
from episode import chooseGreedyAction
import pickle
import matplotlib.style
import matplotlib as mpl
mpl.style.use('bmh')


class GenerateDistribution:

    def __init__(self, numTrials, mctsPolicy, constructTestState, getSavePath,
                 getNNPath, generateModel, generatePolicy,
                 establishFileNameString):
        self.numTrials = numTrials
        self.mctsPolicy = mctsPolicy
        self.constructTestState = constructTestState
        self.getSavePath = getSavePath
        self.getNNPath = getNNPath
        self.generateModel = generateModel
        self.generatePolicy = generatePolicy
        self.establishFileNameString = establishFileNameString

    def __call__(self, df):
        evaluation = OrderedDict()
        evaluation['wolfX'] = df.index.get_level_values('wolfXPosition')[0]
        evaluation['wolfY'] = df.index.get_level_values('wolfYPosition')[0]
        evaluation['sheepX'] = df.index.get_level_values('sheepXPosition')[0]
        evaluation['sheepY'] = df.index.get_level_values('sheepYPosition')[0]
        nnParameter = OrderedDict()
        nnParameter['trainingDataSize'] = df.index.get_level_values(
            'trainingDataSize')[0]
        nnParameter['trainingDataType'] = df.index.get_level_values(
            'trainingDataType')[0]
        nnParameter['batchSize'] = df.index.get_level_values('batchSize')[0]
        nnParameter['trainingStep'] = df.index.get_level_values(
            'trainingStep')[0]
        nnParameter['neuronsPerLayer'] = df.index.get_level_values(
            'neuronsPerLayer')[0]
        nnParameter['sharedLayers'] = df.index.get_level_values(
            'sharedLayers')[0]
        nnParameter['actionLayers'] = df.index.get_level_values(
            'actionLayers')[0]
        nnParameter['valueLayers'] = df.index.get_level_values('valueLayers')[0]
        nnParameter['augmented'] = df.index.get_level_values('augmented')[0]
        saveModelDir = self.getNNPath(nnParameter)
        modelName = self.establishFileNameString(nnParameter)
        modelPath = os.path.join(saveModelDir, modelName)
        model = self.generateModel(
            [nnParameter['neuronsPerLayer']] * nnParameter['sharedLayers'],
            [nnParameter['neuronsPerLayer']] * nnParameter['actionLayers'],
            [nnParameter['neuronsPerLayer']] * nnParameter['valueLayers'])
        if os.path.exists(saveModelDir):
            trainedModel = net.restoreVariables(model, modelPath)
        else:
            print('No Model')
            exit(1)
        nnPolicy = self.generatePolicy(trainedModel)
        saveFileDir = self.getSavePath(nnParameter)
        if not os.path.exists(saveFileDir):
            os.mkdir(saveFileDir)
        saveFileName = establishFileNameString(evaluation)
        saveFilePath = os.path.join(saveFileDir, saveFileName)
        if os.path.exists(saveFilePath):
            print("Data exists {}".format(saveFilePath))
            return None
        testState = self.constructTestState(
            [evaluation['wolfX'], evaluation['wolfY']],
            [evaluation['sheepX'], evaluation['sheepY']])
        if len(testState) == 0:
            zombieDistribution = [0, 0, 0]
            data = [{
                "NNActionDistribution": zombieDistribution,
                'mctsActionDistribution': zombieDistribution
            } for cnt in range(self.numTrials)]
        else:
            evaluateStates = [testState for cnt in range(self.numTrials)]
            mctsActionDistribution = [
                list(self.mctsPolicy(state).values())
                for state in evaluateStates
            ]
            nnActionDistribution = [
                list(nnPolicy(state).values()) for state in evaluateStates
            ]
            data = [{
                "NNActionDistribution": nnAD,
                'mctsActionDistribution': mctsAD
            } for mctsAD, nnAD in zip(mctsActionDistribution,
                                      nnActionDistribution)]
        file = open(saveFilePath, 'wb')
        pickle.dump(data, file)
        file.close()
        return data


def establishFileNameString(parameterDict):
    sortedParameters = sorted(parameterDict.items())
    nameValueStringPairs = [
        parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters
    ]
    nameString = '_'.join(nameValueStringPairs).replace(" ", "")
    return nameString


class DrawHeatMap:

    def __init__(self, groupByVariableNames, subplotIndex, subplotIndexName,
                 extent):
        self.groupByVariableNames = groupByVariableNames
        self.subplotIndex = subplotIndex
        self.subplotIndexName = subplotIndexName
        self.extent = extent

    def __call__(self, dataDF, colName):
        figure = plt.figure(figsize=(12, 10))
        numOfplot = 1
        subDFs = reversed([
            subDF for key, subDF in dataDF.groupby(self.groupByVariableNames[0])
        ])
        OrderedSubDFs = [
            df for subDF in subDFs
            for key, df in subDF.groupby(self.groupByVariableNames[1])
        ]
        for subDF in OrderedSubDFs:
            subplot = figure.add_subplot(self.subplotIndex[0],
                                         self.subplotIndex[1], numOfplot)
            resetDF = subDF.reset_index()[[
                self.subplotIndexName[0], self.subplotIndexName[1], colName
            ]]
            plotDF = resetDF.pivot(index=self.subplotIndexName[1],
                                   columns=self.subplotIndexName[0],
                                   values=colName)
            cValues = plotDF.values
            xticks = plotDF.columns.values
            yticks = plotDF.index.values
            newDf = pd.DataFrame(cValues, index=yticks, columns=xticks)
            image = subplot.imshow(newDf,
                                   vmin=0,
                                   vmax=10,
                                   origin="lower",
                                   extent=self.extent)
            plt.colorbar(image, fraction=0.046, pad=0.04)
            plt.xlabel(self.subplotIndexName[0])
            plt.ylabel(self.subplotIndexName[1])
            # plotDF.plot.scatter(x=self.subplotIndexName[0], y=self.subplotIndexName[1], c=colName, colormap="jet", ax=subplot, vmin=0, vmax=9)
            numOfplot = numOfplot + 1
        plt.suptitle("CrossEntropy Between MCTS and NN")
        plt.subplots_adjust(wspace=0.8, hspace=0.4)


class ConstructTestState:

    def __init__(self, sheepID, wolfID, isTerminal):
        self.sheepID = sheepID
        self.wolfID = wolfID
        self.isTerminal = isTerminal

    def __call__(self, wolfPos, sheepPos):
        unSequentialState = {self.wolfID: wolfPos, self.sheepID: sheepPos}
        state = [
            unSequentialState[key] for key in sorted(unSequentialState.keys())
        ]
        if self.isTerminal(state):
            return []
        return state


def main():
    dataDir = os.path.join(os.pardir, 'data', 'evaluateByCrossEntropy')

    # sample trajectory
    sheepID = 0
    posIndex = [0, 1]
    getSheepPos = state.GetAgentPosFromState(sheepID, posIndex)
    wolfID = 1
    getWolfPos = state.GetAgentPosFromState(wolfID, posIndex)
    killZoneRadius = 25
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)
    degrees = [
        math.pi / 2, 0, math.pi, -math.pi / 2, math.pi / 4, -math.pi * 3 / 4,
        -math.pi / 4, math.pi * 3 / 4
    ]
    establishAction = lambda speed, degree: tuple(
        np.round(speed * transitePolarToCartesian(degree)))
    sheepSpeed = 20
    sheepActionSpace = [
        establishAction(sheepSpeed, degree) for degree in degrees
    ]
    wolfSpeed = sheepSpeed * 0.95
    wolfActionSpace = [establishAction(wolfSpeed, degree) for degree in degrees]
    wolfPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)
    imaginedWolfAction = lambda state: chooseGreedyAction(wolfPolicy(state))
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(
        xBoundary, yBoundary)
    transition = env.TransiteForNoPhysics(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(
        state), [np.array(action), imaginedWolfAction(state)])

    # mcts policy
    cInit = 1
    cBase = 100
    calculateScore = mcts.ScoreChild(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    mctsUniformActionPrior = lambda state: {
        action: 1 / len(sheepActionSpace) for action in sheepActionSpace
    }
    getActionPrior = mctsUniformActionPrior
    initializeChildren = mcts.InitializeChildren(sheepActionSpace,
                                                 sheepTransition,
                                                 getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 10
    rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(
        range(len(sheepActionSpace)))]
    rewardFunction = lambda state, action: 1
    heuristic = lambda state: 0
    estimateValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps,
                                 sheepTransition, rewardFunction, isTerminal,
                                 heuristic)

    numSimulations = 200
    mctsPolicyDistOutput = mcts.MCTS(numSimulations, selectChild, expand,
                                     estimateValue, mcts.backup,
                                     mcts.establishSoftmaxActionDist)

    # pandas
    independentVariables = OrderedDict()
    independentVariables['trainingDataType'] = ['actionDist']
    independentVariables['trainingDataSize'] = [10000, 30000]
    independentVariables['batchSize'] = [2048]
    independentVariables['augmented'] = ['yes', 'no']
    independentVariables['trainingStep'] = [20000]
    independentVariables['neuronsPerLayer'] = [64]
    independentVariables['sharedLayers'] = [3]
    independentVariables['actionLayers'] = [1]
    independentVariables['valueLayers'] = [1]
    wolfDiscreteFactor = 3
    sheepDiscreteFactor = 10
    wolfDiscreteRange = xBoundary[1] / (wolfDiscreteFactor + 1)
    sheepDiscreteRange = xBoundary[1] / (sheepDiscreteFactor + 1)
    independentVariables['wolfXPosition'] = [
        wolfDiscreteRange * (num + 1) for num in range(wolfDiscreteFactor)
    ]
    independentVariables['wolfYPosition'] = [
        wolfDiscreteRange * (num + 1) for num in range(wolfDiscreteFactor)
    ]
    independentVariables['sheepXPosition'] = [
        sheepDiscreteRange * (num + 1) for num in range(sheepDiscreteFactor)
    ]
    independentVariables['sheepYPosition'] = [
        sheepDiscreteRange * (num + 1) for num in range(sheepDiscreteFactor)
    ]
    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=diffIndex)

    constructTestState = ConstructTestState(sheepID=sheepID,
                                            wolfID=wolfID,
                                            isTerminal=isTerminal)
    adDir = "actionDistributions"
    adPath = os.path.join(dataDir, adDir)
    extension = ".pickle"
    fixedParameters = OrderedDict()
    fixedParameters['numSimulations'] = numSimulations
    if not os.path.exists(adPath):
        os.makedirs(adPath)
    getSavePath = GetSavePath(adPath, extension, fixedParameters)
    nnDir = "trainedModel"
    nnPath = os.path.join(dataDir, nnDir)
    nnfixedParameter = {'learningRate': 1e-4}
    getNNPath = GetSavePath(nnPath, '', nnfixedParameter)
    if not os.path.exists(nnPath):
        exit('No Model Dir')
    generateNNPolicy = lambda trainedModel: net.ApproximateActionPrior(
        trainedModel, sheepActionSpace)
    numStateSpace = 4
    numActionSpace = 8
    regularizationFactor = 0
    valueRelativeErrBound = 0.1
    generateModel = net.GenerateModel(numStateSpace, numActionSpace,
                                      regularizationFactor,
                                      valueRelativeErrBound)
    numTrials = 50
    generateDistribution = GenerateDistribution(numTrials, mctsPolicyDistOutput,
                                                constructTestState, getSavePath,
                                                getNNPath, generateModel,
                                                generateNNPolicy,
                                                establishFileNameString)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateDistribution)

    loadData = LoadMultipleTrajectoriesFile(getSavePath, pickle.load)
    computeStatistic = ComputeStatistics(loadData, calculateCrossEntropy)
    statisticDf = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    print(statisticDf)
    # drawingExtent = tuple(xBoundary) + tuple(yBoundary)
    # drawHeatMap = DrawHeatMap(['wolfYPosition', 'wolfXPosition'], [wolfDiscreteFactor, sheepDiscreteFactor], ['sheepXPosition', 'sheepYPosition'], drawingExtent)
    # drawHeatMap(statisticDf, "mean")
    # figureDir = "Graphs"
    # figurePath = os.path.join(dataDir, figureDir)
    # if not os.path.exists(figurePath):
    #     os.makedirs(figurePath)
    # figureName = "Factor{}x{}_{}sample_rollout{}HeatMap.png".format(wolfDiscreteFactor,
    #                                                                 sheepDiscreteFactor,
    #                                                                 numTrials,
    #                                                                 maxRollOutSteps)
    # plt.savefig(os.path.join(figurePath, figureName))


if __name__ == "__main__":
    main()
