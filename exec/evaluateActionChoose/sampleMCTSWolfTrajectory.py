import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel

import mujoco_py as mujoco
import numpy as np




def main():
    # manipulated variables and other important parameters
    sheepId = 0
    wolfId = 1
    killzoneRadius = 2
    numSimulations = 100
    maxRunningSteps = 20
    fixedParameters = {'agentId':sheepId,'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'evaluateActionChoose', 'evaluateTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)


    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])

    actionChooseInMCTS = parametersForTrajectoryPath['chooseActionInMCTS']
    actionChooseInPlay = parametersForTrajectoryPath['chooseActionInPlay']
    agentId = 0
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)
    if not os.path.isfile(trajectorySavePath):
        # Mujoco Environment
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
        # sheepActionInWolfSimulation = lambda state: (0, 0)
        # transitInWolfMCTSSimulation = \
        #     lambda state, wolfSelfAction: transit(state, [sheepActionInWolfSimulation(state), wolfSelfAction])
        numStateSpace = 12
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        sheepModel = generateModel(sharedWidths, actionLayerWidths , valueLayerWidths)

        sheepModelSavePath = os.path.join(dirName, '..', '..', 'data',
                                        'evaluateActionChoose', 'sheepModels',
                                        'agentId=0_depth=4_learningRate=0.001_maxRunningSteps=25_miniBatchSize=512_numSimulations=100_trainSteps=40000')

        restoredSheepModel = restoreVariables(sheepModel, sheepModelSavePath)
        sheepPolicy = ApproximatePolicy(restoredSheepModel, actionSpace)

        if actionChooseInMCTS == 'greedy':
            transitInSheepMCTSSimulation = \
            lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(sheepPolicy(state)),wolfSelfAction])
        else:
            transitInSheepMCTSSimulation = \
            lambda state, wolfSelfAction: transit(state, [sampleAction(sheepPolicy(state)),wolfSelfAction])


        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
        initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation,
                                                            getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = 1/maxRunningSteps
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = -0.1
        maxRolloutSteps = 5
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # sample trajectory
        qPosInit = (0, 0, 0, 0)
        qVelInit = (0, 0, 0, 0)
        qPosInitNoise = 9.7
        qVelInitNoise = 8
        numAgent = 2
        reset = ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)
        if actionChooseInPlay == 'greedy':
            chooseActionMethods = [chooseGreedyAction, chooseGreedyAction]
        else:
            chooseActionMethods = [sampleAction, chooseGreedyAction]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseActionMethods)

        # saving trajectories
        # policy

        policy = lambda state: [sheepPolicy(state), mcts(state)]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        restoredSheepModel.close()


if __name__ == '__main__':
    main()
