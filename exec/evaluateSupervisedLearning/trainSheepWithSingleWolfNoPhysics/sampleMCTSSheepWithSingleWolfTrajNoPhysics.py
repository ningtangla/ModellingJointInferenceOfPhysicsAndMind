import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pickle
import random
import json
import time

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, backup, InitializeChildren, Expand, RollOut
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import chooseGreedyAction, SampleTrajectory
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal

from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle


def main():
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    killzoneRadius = 30
    numSimulations = 100
    maxRunningSteps = 150

    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'trajectoriesWithStillAction')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 2
        sheepId = 0
        wolfId = 1
        positionIndex = [0, 1]

        xBoundary = [0, 600]
        yBoundary = [0, 600]

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)
        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
        isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]

        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        sheepActionSpace.append((0, 0))

        predatorPowerRatio = 2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy
        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolfPolicy(state))])

        # reward function
        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            sheepActionSpace, sheepTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        numActionSpace = len(sheepActionSpace)

        def rolloutPolicy(
            state): return sheepActionSpace[np.random.choice(range(numActionSpace))]

        # rollout
        rolloutHeuristicWeight = 0
        rolloutHeuristic = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getPredatorPos, getPreyPos)
        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                          rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                           rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        finshedTime = time.time() - startTime

        print('timeTaken:', finshedTime)


if __name__ == "__main__":
    main()
