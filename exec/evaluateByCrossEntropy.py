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


class Evaluate:
    def __init__(self, mctsPolicy, nnPolicy, sampleFunction, colName, numOfAgnet, numPosEachState):
        self.mctsPolicy = mctsPolicy
        self.nnPolicy = nnPolicy
        self.sampleFunction = sampleFunction
        self.colName =colName
        self.numPosEachState = numPosEachState
        self.numOfAgent = numOfAgnet

    def __call__(self, df):
        index = df.index[0]
        state = [index[n:n+self.numPosEachState] for n in range(0, len(index), int(len(index)/self.numOfAgent))]
        testState = self.sampleFunction(state)
        if len(testState) == 0:
            return pd.Series({self.colName: 0})
        mctsActionDistribution = [self.mctsPolicy(state) for state in testState]
        nnActionDistribution = [self.nnPolicy([item for sublist in state for item in sublist]) for state in testState]
        crossEntropyList = [calculateCrossEntropy(np.array(list(prediction.values())), np.array(list(target.values()))) for prediction, target in zip(nnActionDistribution, mctsActionDistribution)]
        meanCrossEntropy = np.mean(crossEntropyList)
        return pd.Series({self.colName: meanCrossEntropy})


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
        # plt.show()


class SampleFunction:
    def __init__(self, sampleNum, isTerminal):
        self.sampleNum = sampleNum
        self.isTerminal = isTerminal

    def __call__(self, state):
        return [state for count in range(self.sampleNum) if not self.isTerminal(state)]


def main():
    savePath = "../data/evaluateByCrossEntropy"

    # env
    wolfID = 0
    sheepID = 1
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
    sheepTransition = lambda state, action: transition(state, [wolfDriectChasingPolicy(state), np.array(action)])
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
    modelDir = "../data/evaluateNeuralNetwork/savedModels"
    modelName = "60000data_64x4_minibatch_100kIter_contState_actionDist"
    modelPath = os.path.join(modelDir, modelName)
    generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
    emptyModel = generateModel([64]*4)
    trainedModel = net.restoreVariables(emptyModel, modelPath)
    nnPolicy = net.ApproximateActionPrior(trainedModel, sheepActionSpace)

    # pandas
    wolfDiscreteFactor = 3
    sheepDiscreteFactor = 10
    columnName = "cross_entropy"
    wolfDiscreteRange = xBoundary[1] / (wolfDiscreteFactor+1)
    sheepDiscreteRange = xBoundary[1] / (sheepDiscreteFactor+1)
    samplePoints = 25
    figureName = os.path.join(savePath, "Factor{}x{}_{}sample_rollout{}HeatMap.png".format(wolfDiscreteFactor, sheepDiscreteFactor, samplePoints, maxRollOutSteps))
    wolfXPosition = [wolfDiscreteRange * (i+1) for i in range(wolfDiscreteFactor)]
    wolfYPosition = wolfXPosition
    sheepXPosition = [sheepDiscreteRange * (i+1) for i in range(sheepDiscreteFactor)]
    sheepYPosition = sheepXPosition
    levelValues = [sheepXPosition, sheepYPosition, wolfXPosition, wolfYPosition]
    levelNames = ["sheepXPosition", "sheepYPosition", "wolfXPosition", "wolfYPosition"]
    diffIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=diffIndex)

    sampleFunction = SampleFunction(samplePoints, isTerminal)
    evaluate = Evaluate(mctsPolicy, nnPolicy, sampleFunction, columnName, numOfAgent, numPosEachAgent)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluate)
    drawHeatMap = DrawHeatMap(['wolfYPosition', 'wolfXPosition'], [len(wolfXPosition), len(wolfYPosition)], ['sheepXPosition', 'sheepYPosition'])
    drawHeatMap(resultDF, columnName)
    plt.savefig(figureName)


if __name__ == "__main__":
    main()