import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/sheepWolf")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import policyValueNet as net
import noPhysicsEnv as env
import envSheepChaseWolf
import numpy as np
import pandas as pd
from AnalyticGeometryFunctions import calculateCrossEntropy
from pylab import plt
import policiesFixed
import mcts
import os


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
    savePath = "../data/"
    # env
    wolfID = 0
    sheepID = 1
    posIndex = 0
    numOfAgent = 2
    numPosEachAgent = 2
    numStateSpace = numOfAgent * numPosEachAgent
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    numActionSpace = len(actionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    sheepVelocity = 20
    killZoneRadius = 25
    wolfVelocity = sheepVelocity*0.95

    # mcts policy
    getSheepPos = envSheepChaseWolf.GetAgentPos(sheepID, posIndex, numPosEachAgent)
    getWolfPos = envSheepChaseWolf.GetAgentPos(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policiesFixed.PolicyActionDirectlyTowardsOtherAgent(getWolfPos, getSheepPos, wolfVelocity)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(state, [np.array(action), wolfDriectChasingPolicy(state)])
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    rewardFunction = lambda state, action: 1

    cInit = 1
    cBase = 1
    calculateScore = mcts.CalculateScore(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    getActionPrior = mcts.GetActionPrior(actionSpace)
    initializeChildren = mcts.InitializeChildren(actionSpace, sheepTransition, getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 5
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    heuristic = mcts.HeuristicDistanceToTarget(1, getSheepPos, getWolfPos)
    nodeValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 600
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, nodeValue, mcts.backup, mcts.getSoftmaxActionDist)

    # neuralNetworkModel
    modelPath = os.path.join(savePath, "neuralNetworkGraphVariables/60000data_64x4_minibatch_100kIter_contState_actionDist")
    generateModel = net.GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate=0, regularizationFactor=0, valueRelativeErrBound=0.0)
    emptyModel = generateModel([64]*4)
    trainedModel = net.restoreVariables(emptyModel, modelPath)
    nnPolicy = net.ApproximateActionPrior(trainedModel, actionSpace)

    # pandas
    wolfDiscreteFactor = 3
    sheepDiscreteFactor = 10
    columnName = "cross_entropy"
    wolfDiscreteRange = xBoundary[1] / (wolfDiscreteFactor+1)
    sheepDiscreteRange = xBoundary[1] / (sheepDiscreteFactor+1)
    samplePoints = 25
    figureName = os.path.join(savePath, "evaluateByCrossEntropy/Factor{}x{}_{}sample_HeatMap.png".format(wolfDiscreteFactor, sheepDiscreteFactor, samplePoints))
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