import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import policyValueNet as net
import envNoPhysics as env
import wrapperFunctions
import numpy as np
import pandas as pd
from measurementFunctions import calculateCrossEntropy
from analyticGeometryFunctions import transitePolarToCartesian
from pylab import plt
import policies
import mcts
import os
import math
from collections import OrderedDict
from evaluationFunctions import GetSavePath, ComputeStatistics, LoadTrajectories
import pickle


class GenerateDistribution:
    def __init__(self, mctsPolicy, nnPolicy, constructTestState, getSavePath, numTrials):
        self.mctsPolicy = mctsPolicy
        self.nnPolicy = nnPolicy
        self.constructTestState = constructTestState
        self.numTrials = numTrials
        self.getSavePath = getSavePath

    def __call__(self, df):
        wolfX = df.index.get_level_values('wolfXPosition')[0]
        wolfY = df.index.get_level_values('wolfYPosition')[0]
        sheepX = df.index.get_level_values('sheepXPosition')[0]
        sheepY = df.index.get_level_values('sheepYPosition')[0]
        indexLevelNames = df.index.names
        parameters = {levelName: df.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        saveFileName = self.getSavePath(parameters)
        if os.path.exists(saveFileName):
            print("Data exists {}".format(saveFileName))
            return None
        testState = self.constructTestState([wolfX, wolfY], [sheepX, sheepY])
        if len(testState) == 0:
            zombieDistribution = [1 / 8 for cnt in range(8)]
            data = [{"NNActionDistribution": zombieDistribution, 'mctsActionDistribution': zombieDistribution} for cnt in range(self.numTrials)]
        else:
            evaluateStates = [testState for cnt in range(self.numTrials)]
            mctsActionDistribution = [list(self.mctsPolicy(state).values()) for state in evaluateStates]
            nnActionDistribution = [list(self.nnPolicy(state).values()) for state in evaluateStates]
            data = [{"NNActionDistribution": nnAD, 'mctsActionDistribution': mctsAD} for mctsAD, nnAD in zip(mctsActionDistribution, nnActionDistribution)]
        file = open(saveFileName, 'wb')
        pickle.dump(data, file)
        file.close()
        return data


class DrawHeatMap:
    def __init__(self, groupByVariableNames, subplotIndex, subplotIndexName):
        self.groupByVariableNames = groupByVariableNames
        self.subplotIndex = subplotIndex
        self.subplotIndexName = subplotIndexName

    def __call__(self, dataDF, colName):
        figure = plt.figure(figsize=(12, 10))
        numOfplot = 1
        subDFs = reversed([subDF for key, subDF in dataDF.groupby(self.groupByVariableNames[0])])
        OrderedSubDFs = [df for subDF in subDFs for key, df in subDF.groupby(self.groupByVariableNames[1])]
        for subDF in OrderedSubDFs:
            subplot = figure.add_subplot(self.subplotIndex[0], self.subplotIndex[1], numOfplot)
            plotDF = subDF.reset_index()
            plotDF.plot.scatter(x=self.subplotIndexName[0], y=self.subplotIndexName[1], c=colName, colormap="jet", ax=subplot, vmin=0, vmax=9)
            numOfplot = numOfplot + 1
        plt.subplots_adjust(wspace=0.8, hspace=0.4)


class ConstructTestState:
    def __init__(self, sheepID, wolfID, isTerminal):
        self.sheepID = sheepID
        self.wolfID = wolfID
        self.isTerminal = isTerminal

    def __call__(self, wolfPos, sheepPos):
        unSequentialState = {self.wolfID:wolfPos, self.sheepID: sheepPos}
        state = [unSequentialState[key] for key in sorted(unSequentialState.keys())]
        if self.isTerminal(state):
            return []
        return state


def main():
    dataDir = "../data/evaluateByCrossEntropy"
    # env
    wolfID = 1
    sheepID = 0
    posIndex = 0
    numOfAgent = 2
    numPosEachAgent = 2
    numStateSpace = numOfAgent * numPosEachAgent
    numActionSpace = 8
    sheepSpeed = 20
    wolfSpeed = sheepSpeed * 0.95
    degrees = [math.pi/2, 0, math.pi, -math.pi/2, math.pi/4, -math.pi*3/4, -math.pi/4, math.pi*3/4]
    sheepActionSpace = [tuple(np.round(sheepSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    wolfActionSpace = [tuple(np.round(wolfSpeed * transitePolarToCartesian(degree))) for degree in degrees]
    print(sheepActionSpace)
    print(wolfActionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    killZoneRadius = 25

    # mcts policy
    getSheepPos = wrapperFunctions.GetAgentPosFromState(sheepID, posIndex, numPosEachAgent)
    getWolfPos = wrapperFunctions.GetAgentPosFromState(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getWolfPos, getSheepPos)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(state, [np.array(action), wolfDriectChasingPolicy(state)])
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    rewardFunction = lambda state, action: 1

    cInit = 1
    cBase = 1
    calculateScore = mcts.CalculateScore(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    mctsUniformActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}
    getActionPrior = mctsUniformActionPrior
    initializeChildren = mcts.InitializeChildren(sheepActionSpace, sheepTransition, getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 10
    rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
    heuristic = lambda state: 0
    nodeValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 600
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, nodeValue, mcts.backup, mcts.establishSoftmaxActionDist)

    # neuralNetworkModel
    modelDir = "savedModels"
    modelName = "60000data_64x4_minibatch_100kIter_contState_actionDist"
    modelPath = os.path.join(dataDir, modelDir, modelName)
    if not os.path.exists(modelPath):
        print("Model {} does not exist".format(modelPath))
        exit(1)
    generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
    emptyModel = generateModel([64]*4)
    trainedModel = net.restoreVariables(emptyModel, modelPath)
    nnPolicy = net.ApproximateActionPrior(trainedModel, sheepActionSpace)

    # pandas
    wolfDiscreteFactor = 3
    sheepDiscreteFactor = 10
    wolfDiscreteRange = xBoundary[1] / (wolfDiscreteFactor+1)
    sheepDiscreteRange = xBoundary[1] / (sheepDiscreteFactor+1)
    numTrials = 25
    wolfXPosition = [wolfDiscreteRange * (i+1) for i in range(wolfDiscreteFactor)]
    wolfYPosition = wolfXPosition
    sheepXPosition = [sheepDiscreteRange * (i+1) for i in range(sheepDiscreteFactor)]
    sheepYPosition = sheepXPosition
    levelValues = [sheepXPosition, sheepYPosition, wolfXPosition, wolfYPosition]
    levelNames = ["sheepXPosition", "sheepYPosition", "wolfXPosition", "wolfYPosition"]
    diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=diffIndex)

    constructTestState = ConstructTestState(sheepID=sheepID, wolfID=wolfID, isTerminal=isTerminal)
    adDir = "actionDistributions"
    adPath = os.path.join(dataDir, adDir)
    extension = ".pickle"
    fixedParameters = OrderedDict()
    fixedParameters['numSimulations'] = numSimulations
    if not os.path.exists(adPath):
        os.makedirs(adPath)
    getSavePath = GetSavePath(adPath, extension, fixedParameters)
    generateDistribution = GenerateDistribution(mctsPolicy, nnPolicy, constructTestState, getSavePath, numTrials)
    resultDF = toSplitFrame.groupby(levelNames).apply(generateDistribution)

    loadData = LoadTrajectories(getSavePath)
    computeStatistic = ComputeStatistics(loadData, numTrials, calculateCrossEntropy)
    statisticDf = toSplitFrame.groupby(levelNames).apply(computeStatistic)
    drawHeatMap = DrawHeatMap(['wolfYPosition', 'wolfXPosition'], [len(wolfXPosition), len(wolfYPosition)], ['sheepXPosition', 'sheepYPosition'])
    drawHeatMap(statisticDf, "mean")
    figureDir = "Graphs"
    figurePath = os.path.join(dataDir, figureDir)
    if not os.path.exists(figurePath):
        os.makedirs(figurePath)
    figureName = "Factor{}x{}_{}sample_rollout{}HeatMap.png".format(wolfDiscreteFactor,
                                                                    sheepDiscreteFactor,
                                                                    numTrials,
                                                                    maxRollOutSteps)
    plt.savefig(os.path.join(figurePath, figureName))


if __name__ == "__main__":
    main()