import sys
sys.path.append("..")

from src.constrainedChasingEscapingEnv.envNoPhysics import Reset

sys.path.append('..')
import numpy as np
import pickle

import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.policies as policies
import src.constrainedChasingEscapingEnv.wrapperFunctions as wrapperFunctions
import src.algorithms.mcts as mcts
import src.play as play
from exec.evaluationFunctions import GetSavePath


def main():
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
    killZoneRadius = 5
    wolfVelocity = sheepVelocity * 0.95

    getSheepPos = wrapperFunctions.GetAgentPosFromState(sheepID, posIndex, numPosEachAgent)
    getWolfPos = wrapperFunctions.GetAgentPosFromState(wolfID, posIndex, numPosEachAgent)
    checkBoundaryAndAdjust = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    wolfDriectChasingPolicy = policies.HeatSeekingDiscreteDeterministicPolicy(actionSpace, getSheepPos, getWolfPos)
    transition = env.TransitionForMultiAgent(checkBoundaryAndAdjust)
    sheepTransition = lambda state, action: transition(np.array(state), [wolfDriectChasingPolicy(state), np.array(action)])

    initPosition = np.array([[20, 20], [30, 30]])
    initNoise = [0, 0]
    reset = env.Reset(numOfAgent, initPosition, initNoise)
    isTerminal = env.IsTerminal(getWolfPos, getSheepPos, killZoneRadius)

    rewardFunction = lambda state, action: 1

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
    heuristic = lambda state: 0
    estimateValue = mcts.RollOut(rolloutPolicy, maxRollOutSteps, sheepTransition, rewardFunction, isTerminal, heuristic)

    numSimulations = 200
    mctsPolicy = mcts.MCTS(numSimulations, selectChild, expand, estimateValue, mcts.backup, mcts.selectGreedyAction)
    mctsPolicyDistOutput = mcts.MCTS(numSimulations, selectChild, expand, estimateValue, mcts.backup, mcts.establishSoftmaxActionDist)

    # sample trajectories
    maxRunningSteps = 100

    sampleTraj = play.SampleTrajectory(maxRunningSteps, transition, isTerminal, reset)
    policy = lambda state: [wolfDriectChasingPolicy(state), mctsPolicy(state)]

    agentDist2Action = play.agentDistToGreedyAction
    worldDist2Action = lambda worldDist: play.worldDistToAction(agentDist2Action, worldDist)
    sampleTrajWithActionDist = play.SampleTrajectoryWithActionDist(maxRunningSteps, transition, isTerminal, reset, worldDist2Action)
    policyDistOutput = lambda state: [wolfDriectChasingPolicy(state), mctsPolicyDistOutput(state)]

    useActionDist = False
    numTrajs = 1
    if not useActionDist:
        trajs = [sampleTraj(policy) for _ in range(numTrajs)]
    else:
        trajs = [sampleTrajWithActionDist(policyDistOutput) for _ in range(numTrajs)]

    saveOn = True
    savePath = "temp.pkl"  # TODO: use GetSavePath
    if saveOn:
        with open(savePath, "wb") as f:
            pickle.dump(trajs, f)


if __name__ == "__main__":
    main()
