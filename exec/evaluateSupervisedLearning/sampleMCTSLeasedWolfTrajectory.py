import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel

import mujoco_py as mujoco
import numpy as np




def main():
    # manipulated variables and other important parameters
    killzoneRadius = 2
    numSimulations = 100
    maxRunningSteps = 25
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'evaluateSupervisedLearning', 'Leasedtrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)


    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])

    agentId = int(parametersForTrajectoryPath['agentId'])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    if not os.path.isfile(trajectorySavePath):
        # Mujoco Environment
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'leased.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
        transitInWolfMCTSSimulation = \
            lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(randomPolicy(state)), wolfSelfAction, chooseGreedyAction(randomPolicy(state)])

        # WolfActionInSheepSimulation = lambda state: (0, 0)
        # transitInSheepMCTSSimulation = \
        #     lambda state, sheepSelfAction: transit(state, [sheepSelfAction, WolfActionInSheepSimulation(state)])

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
        initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTSSimulation,
                                                            getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = -0.05
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = 0.1
        maxRolloutSteps = 10
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # sample trajectory
        qPosInit = (0, ) * 24
        qVelInit = (0, ) * 24
        qPosInitNoise = 9.7
        qVelInitNoise = 8
        numAgent = 2
        reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # saving trajectories
        # policy
        policy = lambda state: [randomPolicy(state), mcts(state), randomPolicy(state)]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
