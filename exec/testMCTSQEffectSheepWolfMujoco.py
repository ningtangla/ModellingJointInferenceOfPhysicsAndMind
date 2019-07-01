import numpy as np
import pandas as pd
import pickle
import pylab as plt
from matplotlib import pyplot as plt
from datetime import datetime
from collections import OrderedDict
import os
import skvideo.io
import copy

skvideo.setFFmpegPath(os.path.join("usr", "local", "bin"))

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, HeuristicDistanceToTarget
import envMujoco as env
import reward
from envSheepChaseWolf import stationaryWolfPolicy, WolfPolicyForceDirectlyTowardsSheep, DistanceBetweenActualAndOptimalNextPosition, \
    GetTrialTrajectoryFromDf, GetPosFromTrajectory, GetAgentPos
from envMujoco import Reset, TransitionFunction, IsTerminal
from play import SampleTrajectory


def drawPerformanceLine(dataDf, axForDraw, xLabel, legendLabels):                                                          # Line
    for key, grp in dataDf.groupby(level=['rolloutHeuristicWeight', 'maxRolloutSteps'], group_keys=False):                                     # remove the reset_index
        grp.index = grp.index.droplevel(level=[1, 2])
        print(grp)
        grp.plot(ax=axForDraw, kind='line', y='mean', yerr='std', label = legendLabels[key], marker='o')


class RunTrial:
    def __init__(self, getRolloutHeuristic, getRollout, getMCTS, sampleTrajectory):  # break this line
        self.sampleTrajectory = sampleTrajectory
        self.getMCTS = getMCTS
        self.getRollout = getRollout
        self.getRolloutHeuristic = getRolloutHeuristic

    def __call__(self, conditionDf):
        modelDf = conditionDf.reset_index()
        rolloutHeuristicWeight = modelDf['rolloutHeuristicWeight'][0]                                                   # reset_index and use column values. Maybe reset outside
        maxRolloutSteps = modelDf['maxRolloutSteps'][0]
        numSimulations = modelDf['numSimulations'][0]

        rolloutHeuristic = self.getRolloutHeuristic(rolloutHeuristicWeight)
        rollout = self.getRollout(maxRolloutSteps, rolloutHeuristic)
        sheepPolicyMCTS = self.getMCTS(numSimulations, rollout)
        allAgentsPolicies = [sheepPolicyMCTS, stationaryWolfPolicy]
        policy = lambda state: [agentPolicy(state) for agentPolicy in allAgentsPolicies]

        trajectory = self.sampleTrajectory(policy)
        trajectorySeries = pd.Series({'trajectory': trajectory})

        return trajectorySeries


def main(): # comments for explanation
    startTime = datetime.now()

    # experiment conditions
    numTrials = 2
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [5, 6, 7]
    manipulatedVariables['rolloutHeuristicWeight'] = [0.1, 0]
    manipulatedVariables['maxRolloutSteps'] = [10, 0]
    manipulatedVariables['trialIndex'] = list(range(numTrials))

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    toSplitFrame = toSplitFrame.select(lambda row: (row[1] == 0.1 and row[2] == 0) or (row[1] == 0 and row[2] == 10), axis = 0)

    # reset function
    envModelName = 'twoAgents'
    qPosInit = [-9.8, 9.8, 8, 0]
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    # isTerminal
    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius)

    # transit for multiplayer transition
    # sheepTransit for sheep's MCTS simulation (only takes sheep's action as input)
    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationFrames)
    sheepTransit = lambda state, action: transit(state, [action, stationaryWolfPolicy(state)])

    # reward function (aliveBonus is negative and deathPenalty is positive because sheep is chasing wolf).
    aliveBonus = -0.05
    deathPenalty = 1
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
    maxRunningSteps = 5

    getSheepXPos = GetAgentPos(sheepId, xPosIndex, numXPosEachAgent)
    getWolfXPos = GetAgentPos(wolfId, xPosIndex, numXPosEachAgent)

    # wrapper functions
    getRolloutHeuristic = lambda rolloutHeuristicWeight: HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)    # change the interface of heuristic function
    getRollout = lambda maxRolloutSteps, rolloutHeuristic: RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal,
                          rolloutHeuristic)

    getMCTS = lambda numSimulations, rollout: MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)

    # sample trajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)

    runTrial = RunTrial(getRolloutHeuristic, getRollout, getMCTS, sampleTrajectory)

    # run all trials and generate trajectories
    trialDataDf = toSplitFrame.groupby(levelNames).apply(runTrial)
    print(trialDataDf.values[0][0])
    pickle_out = open('trajectories/testQEffectOnMCTSMujoco.pickle', 'wb')
    pickle.dump(trialDataDf, pickle_out)
    pickle_out.close()

    # meausurement function
    initState = reset()
    allAgentsOptimalActions = [(10, 0), (0, 0)]     # sheep directly moves towards wolf
    optimalNextState = transit(initState, allAgentsOptimalActions)
    optimalNextPosition = getSheepXPos(optimalNextState)
    nextPositionIndex = 1   # we are interested in the state at 2nd time step
    stateIndexInTuple = 0   # tuple = (state, action). State is the first element.
    firstTrialIndex = 0
    getSheepXPosAtNextStepFromTrajectory = GetPosFromTrajectory(nextPositionIndex, stateIndexInTuple, sheepId, xPosIndex,
                                                          numXPosEachAgent)
    getFirstTrajectoryFromDf = GetTrialTrajectoryFromDf(firstTrialIndex)
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(optimalNextPosition,
                                                                      getSheepXPosAtNextStepFromTrajectory,
                                                                      getFirstTrajectoryFromDf)

    # make measurements on individual trial
    measurementDf = trialDataDf.groupby(levelNames).apply(measurementFunction)
    pickle_out = open('measurements/testQEffectOnMCTSMujoco.pickle', 'wb')
    pickle.dump(measurementDf, pickle_out)
    pickle_out.close()

    # compute statistics on the measurements (mean, standard error)
    manipulatedVariablesExceptTrialIndex = copy.deepcopy(levelNames)
    manipulatedVariablesExceptTrialIndex.remove('trialIndex')
    dataStatisticsDf = measurementDf.groupby(manipulatedVariablesExceptTrialIndex).distance.agg(['mean', 'std'])
    print(dataStatisticsDf)
    pickle_out = open('statistics/testQEffectOnMCTSMujoco.pickle', 'wb')
    pickle.dump(dataStatisticsDf, pickle_out)
    pickle_out.close()

    # plot the statistics
    fig = plt.figure()

    plotRowNum = 1
    plotColNum = 1
    plotCounter = 1

    legendLabels = {(0, 10): 'Rollout', (0.1, 0): 'No Rollout'}

    axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
    drawPerformanceLine(dataStatisticsDf, axForDraw, 'numSimulations', legendLabels)

    plt.legend(loc = 'best')

    plt.show()

    endTime = datetime.now()
    print("Time taken: ", endTime-startTime)


if __name__ == "__main__":
    main()
