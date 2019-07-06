import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import ScoreChild, SelectChild, MCTS, InitializeChildren, Expand, backup, establishPlainActionDist
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateActionPrior, ApproximateValueFunction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy

from collections import OrderedDict
import pandas as pd
from mujoco_py import load_model_from_path, MjSim
from time import time
from matplotlib import pyplot as plt


class PreparePolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, numSimulations):
        sheepPolicy = self.getSheepPolicy(numSimulations)
        wolfPolicy = self.getWolfPolicy(numSimulations)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class RunEvaluationForCondition:
    def __init__(self, numTrials, sampleTrajectory, preparePolicy):
        self.numTrials = numTrials
        self.sampleTrajectory = sampleTrajectory
        self.preparePolicy = preparePolicy

    def __call__(self, oneConditionDf):
        numSimulations = oneConditionDf.index.get_level_values('numSimulations')[0]
        policy = self.preparePolicy(numSimulations)
        startTime = time()
        trajectories = [self.sampleTrajectory(policy) for _ in range(self.numTrials)]
        endTime = time()
        meanTime = (endTime - startTime)/self.numTrials
        returnSeries = pd.Series({'time': meanTime})

        return returnSeries


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [1, 2, 4, 8, 16, 32, 64, 128]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = load_model_from_path(physicsDynamicsPath)
    physicsSimulation = MjSim(physicsModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    wolfActionInSheepMCTSSimulation = lambda state: (0, 0)
    transitInSheepMCTSSimulation = lambda state, sheepSelfAction: transit(state, [sheepSelfAction,
                                                                                  wolfActionInSheepMCTSSimulation(state)])

    # MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # NN Model
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    NNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # functions to use NN Model predictions
    approximateActionPrior = ApproximateActionPrior(NNModel, actionSpace)
    getStateFromNode = lambda node: list(node.id.values())[0]
    approximateValue = ApproximateValueFunction(NNModel)
    approximateValueFromNode = lambda node: approximateValue(getStateFromNode(node))

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    # wrapper for expand
    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation, approximateActionPrior)
    expandNNPrior = Expand(isTerminal, initializeChildrenNNPrior)

    getMCTSNN = lambda numSimulations: MCTS(numSimulations, selectChild, expandNNPrior, approximateValueFromNode, backup,
                                            establishPlainActionDist)

    # all agents' policies
    getWolfPolicy = lambda numSimulations: stationaryAgentPolicy
    preparePolicy = PreparePolicy(getMCTSNN, getWolfPolicy)

    # sample trajectory
    maxRunningSteps = 10
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    # run evaluation for each condition
    numTrials = 20
    runEvaluationForCondition = RunEvaluationForCondition(numTrials, sampleTrajectory, preparePolicy)
    resultDf = toSplitFrame.groupby(levelNames).apply(runEvaluationForCondition)

    # plot
    fig = plt.figure()
    numColumns = 1
    numRows = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    resultDf.plot(ax=axForDraw, y='time', title='Time taken to run {} steps of MCTS (averaged over {} trials)'.
                  format(maxRunningSteps, numTrials), marker='o')
    plt.ylabel("time in seconds")
    plt.show()


if __name__ == '__main__':
    main()


