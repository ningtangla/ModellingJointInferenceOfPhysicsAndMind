import numpy as np
import pandas as pd
import pickle
import pylab as plt
from matplotlib import pyplot as plt
from datetime import datetime
from collections import OrderedDict
import skvideo.io
import os

skvideo.setFFmpegPath(os.path.join("usr", "local", "bin"))

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, HeuristicDistanceToTarget
import envMujoco as env
import reward
from envSheepChaseWolf import stationaryWolfPolicy, WolfPolicyForceDirectlyTowardsSheep, DistanceBetweenActualAndOptimalNextPosition, \
    GetAgentPosFromTrajectory, GetAgentPos, GetAgentActionFromTrajectoryDf
from envMujoco import Reset, TransitionFunction, IsTerminal
from play import SampleTrajectory


def drawHistogram(dataDf, axForDraw):                                                               # Line
    print('dataDf', dataDf)
    countDf = dataDf['action'].value_counts()
    countDf.plot.bar(ax=axForDraw, rot='horizontal')


class RunTrial:
    def __init__(self, getReset, getSampleTrajectory, policy):  # break this line
        self.getReset = getReset
        self.getSampleTrajectory = getSampleTrajectory
        self.policy = policy

    def __call__(self, conditionDf):
        modelDf = conditionDf.reset_index()
        yCoordinate = modelDf['yCoordinate'][0]                                                   # reset_index and use column values. Maybe reset outside

        reset = self.getReset(yCoordinate)
        sampleTrajectory = self.getSampleTrajectory(reset)

        trajectory = sampleTrajectory(self.policy)
        trajectorySeries = pd.Series({'trajectory': trajectory})

        return trajectorySeries


def main(): # comments for explanation
    startTime = datetime.now()

    # experiment conditions
    maxRunningSteps = 1
    numTrials = 200
    numSimulations = 200
    manipulatedVariables = OrderedDict()
    manipulatedVariables['yCoordinate'] = [-9.75, -5, 0, 5, 9.75]
    manipulatedVariables['trialIndex'] = list(range(numTrials))             # this is not being manipulated

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # reset function
    envModelName = 'twoAgents'
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    getQPosFromYCoordinate = lambda yCoordinate: [-9.75, yCoordinate, -9.15, yCoordinate]

    # isTerminal
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius)

    # transit for multiplayer transition
    # sheepTransit for sheep's MCTS simulation (only takes sheep's action as input)
    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryWolfPolicy(state)])

    # reward function
    aliveBonus = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    # select child
    cInit = 1
    cBase = 100
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    getActionPrior = GetActionPrior(actionSpace)

    # initialize children; expand
    initializeChildren = InitializeChildren(actionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    # select next action
    selectNextAction = SelectNextAction(sheepTransit)

    sheepId = 0
    wolfId = 1
    xPosIndex = 2
    numXPosEachAgent = 2

    getSheepXPos = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

    # rollout
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # wrapper function
    getReset = lambda yCoordinate: Reset(envModelName, getQPosFromYCoordinate(yCoordinate), qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # All agents' policies
    sheepPolicyMCTS = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)
    wolfActionMagnitude = 5
    wolfPolicyDirectlyTowardsSheep = WolfPolicyForceDirectlyTowardsSheep(getSheepXPos, getWolfXPos, wolfActionMagnitude)
    allAgentsPolicies = [sheepPolicyMCTS, wolfPolicyDirectlyTowardsSheep]
    policy = lambda state: [agentPolicy(state) for agentPolicy in allAgentsPolicies]

    # sample trajectory
    getSampleTrajectory = lambda reset: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    runTrial = RunTrial(getReset, getSampleTrajectory, policy)

    # run all trials and generate trajectories
    trialDataDf = toSplitFrame.groupby(levelNames).apply(runTrial)                                                      # call this save trajectory (also take the number of trials for this condition)
    pickle_out = open('trajectories/testSheepFirstStepEscape.pickle', 'wb')                                             # save as csv instead of pickle
    pickle.dump(trialDataDf, pickle_out)                                                                                # save each condition in a separate file
    pickle_out.close()                                                                                                  # keep appending

    print('trialDataDf')
    print(trialDataDf)

    # meausurement function
    timeStep = 0
    getSheepActionFromAllAgentsActions = lambda allAgentActions: allAgentActions[sheepId]
    getFirstTrajectoryFromDf = lambda trajectoryDf: trajectoryDf.values[0][0]
    getAllAgentActionsFromTrajectory = lambda trajectory, timeStep: trajectory[timeStep][1]
    getSheepFirstActionFromTrajectoryDf = GetAgentActionFromTrajectoryDf(getFirstTrajectoryFromDf, timeStep, getSheepActionFromAllAgentsActions, getAllAgentActionsFromTrajectory)

    # make measurements on individual trial
    measurementDf = trialDataDf.groupby(levelNames).apply(getSheepFirstActionFromTrajectoryDf)
    pickle_out = open('measurements/testQEffectOnMCTSMujoco.pickle', 'wb')
    pickle.dump(measurementDf, pickle_out)
    pickle_out.close()

    print('measurementDf')
    print(measurementDf)

    # # compute statistics on the measurements (mean, standard error)
    dataStatisticsDf = measurementDf.groupby('yCoordinate')
    # pickle_out = open('statistics/testSheepFirstStepEscape.pickle', 'wb')
    # pickle.dump(dataStatisticsDf, pickle_out)
    # pickle_out.close()

    # plot the statistics
    fig = plt.figure()

    plotRowNum = 5
    plotColNum = 1
    plotCounter = 1

    for (key, dataDf) in dataStatisticsDf:
        axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
        drawHistogram(dataDf, axForDraw)
        plotCounter += 1

    plt.legend(loc = 'best')

    plt.show()

    endTime = datetime.now()
    print("Time taken: ", endTime-startTime)


if __name__ == "__main__":
    main()


