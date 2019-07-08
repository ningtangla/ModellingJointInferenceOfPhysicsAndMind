import sys
sys.path.append("../src/neuralNetwork")
sys.path.append("../src/constrainedChasingEscapingEnv")
sys.path.append("../src/algorithms")
sys.path.append("../src")
import os
import numpy as np
import pickle

import envNoPhysics as env
import policies
import wrapperFunctions
import mcts
import play
from evaluationFunctions import GetSavePath


def main():
    # env
    sheepID = 0
    wolfID = 1
    posIndex = 0
    numOfAgent = 2
    numPosEachAgent = 2
    actionSpace = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    numActionSpace = len(actionSpace)
    xBoundary = [0, 180]
    yBoundary = [0, 180]
    killZoneRadius = 5

    getSheepPos = wrapperFunctions.GetAgentPosFromState(sheepID, posIndex, numPosEachAgent)
    getWolfPos = wrapperFunctions.GetAgentPosFromState(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(actionSpace, getWolfPos, getSheepPos)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state), [np.array(action), wolfDriectChasingPolicy(state)])

    initPosition = np.array([[30, 30], [20, 20]])
    initNoise = [0, 0]
    reset = env.Reset(numOfAgent, initPosition, initNoise)
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    # mcts policy
    cInit = 1
    cBase = 1
    calculateScore = mcts.CalculateScore(cInit, cBase)
    selectChild = mcts.SelectChild(calculateScore)

    mctsUniformActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}
    getActionPrior = mctsUniformActionPrior
    initializeChildren = mcts.InitializeChildren(actionSpace, sheepTransition, getActionPrior)
    expand = mcts.Expand(isTerminal, initializeChildren)

    maxRollOutSteps = 10
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rewardFunction = lambda state, action: 1
    heuristic = lambda state: 0
    estimateValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 200
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, estimateValue, mcts.backup, mcts.selectGreedyAction)
    mctsPolicyDistOutput = mcts.MCTS(numSimulations, selectChild, expand, estimateValue, mcts.backup, mcts.establishSoftmaxActionDist)

    # sample trajectories
    maxRunningSteps = 30
    agentDist2Action = play.agentDistToGreedyAction
    worldDist2Action = lambda worldDist: play.worldDistToAction(agentDist2Action, worldDist)
    sampleTrajWithActionDist = play.SampleTrajectoryWithActionDist(maxRunningSteps, transition, isTerminal, reset, worldDist2Action)
    policyDistOutput = lambda state: [mctsPolicyDistOutput(state), wolfDriectChasingPolicy(state)]

    numTrajs = 200
    trajs = [sampleTrajWithActionDist(policyDistOutput) for _ in range(numTrajs)]
    print("Avg traj length = {}".format(np.mean([len(traj) for traj in trajs])))

    dataDirectory = '../data/trainingDataForNN/trajectories'
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    extension = '.pickle'
    getSavePath = GetSavePath(dataDirectory, extension)
    varDict = {}
    varDict["initPos"] = list(initPosition.flatten())
    varDict["rolloutSteps"] = maxRollOutSteps
    varDict["numSimulations"] = numSimulations
    varDict["maxRunningSteps"] = maxRunningSteps
    varDict["numTrajs"] = numTrajs
    savePath = getSavePath(varDict)

    saveOn = True
    if saveOn:
        with open(savePath, "wb") as f:
            pickle.dump(trajs, f)
        print("Saved trajectories in {}".format(savePath))


if __name__ == "__main__":
    main()
