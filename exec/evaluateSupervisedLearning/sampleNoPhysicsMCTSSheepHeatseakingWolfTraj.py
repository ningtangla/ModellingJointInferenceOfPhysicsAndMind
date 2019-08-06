import sys
import os
import json
import numpy as np
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset,StayInBoundaryByReflectVelocity
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, backup, RollOut


from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy,HeatSeekingDiscreteDeterministicPolicy





def main():
    # manipulated variables and other important parameters
    killzoneRadius = 35
    numSimulations = 200
    maxRunningSteps = 100
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'evaluateSupervisedLearningNoPhyscis', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    # debug
    # startSampleIndex = 1
    # endSampleIndex = 10
    # trajectorySavePath = generateTrajectorySavePath({'sampleIndex':(startSampleIndex, endSampleIndex)})
    if not os.path.isfile(trajectorySavePath):
        # Mujoco Environment
        originalActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

        preySpeedRatio = 3
        actionSpace = list(map(tuple, np.array(originalActionSpace) * preySpeedRatio))
        numActionSpace = len(actionSpace)
        predatorSpeedRatio = 2.4
        wolfActionSpace = list(map(tuple, np.array(originalActionSpace) * predatorSpeedRatio))

        xBoundary = [0, 320]
        yBoundary = [0, 240]
        numOfAgent = 2
        sheepId = 0
        wolfId = 1
        positionIndex = [0, 1]
        getSheepPos = GetAgentPosFromState(sheepId, positionIndex)
        getWolfPos = GetAgentPosFromState(wolfId, positionIndex)
        isTerminal = IsTerminal(killzoneRadius, getWolfPos, getSheepPos)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        isTerminal = IsTerminal(getWolfPos, getSheepPos, killzoneRadius)
        transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        woldPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getWolfPos, getSheepPos, computeAngleBetweenVectors)

        WolfActionInSheepSimulation = lambda state: chooseGreedyAction(woldPolicy(state))
        transitInSheepMCTSSimulation = \
            lambda state, sheepSelfAction: transit(state, [sheepSelfAction, WolfActionInSheepSimulation(state)])


        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
        initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation, getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = 1/maxRunningSteps
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = -0.1
        maxRolloutSteps = 10
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfPos, getSheepPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal, rolloutHeuristic)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # sample trajectory
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # policy
        policy = lambda state: [mcts(state), woldPolicy(state)]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        # print([len(tra) for tra in trajectories], np.mean([len(tra) for tra in trajectories]))
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
