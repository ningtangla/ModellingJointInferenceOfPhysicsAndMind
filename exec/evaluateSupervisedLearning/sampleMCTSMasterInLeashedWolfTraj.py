import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, ResetUniformForLeashed,TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, TransitionFunction
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy,HeatSeekingDiscreteDeterministicPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
import mujoco_py as mujoco
import numpy as np




def main():
    # manipulated variables and other important parameters
    killzoneRadius = 2
    numSimulations = 200
    maxRunningSteps = 25
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','evaluateSupervisedLearning', 'leashedMasterTrajectories')

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
        preyPowerRatio = 1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.4
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
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

# neural network init
        numStateSpace = 18
        numActionSpace = len(masterActionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        depth = 4
        initWolfNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initSheepNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace)

# sheep NN model
        sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedSheepNNModels','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        sheepPreTrainModel = restoreVariables(initSheepNNModel, sheepPreTrainModelPath)
        sheepPolicy = ApproximatePolicy(sheepPreTrainModel, sheepActionSpace)

        transitInMasterMCTSSimulation = \
            lambda state, masterSelfAction: transit(state, [sampleAction(sheepPolicy(state[:3])),sampleAction(wolfPolicy(state[:3])), masterSelfAction])

# MCTS master
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in masterActionSpace}
        initializeChildrenUniformPrior = InitializeChildren(masterActionSpace, transitInMasterMCTSSimulation, getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = 0.05
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: masterActionSpace[np.random.choice(range(numActionSpace))]
        maxRolloutSteps = 10
        rolloutHeuristicWeight = deathPenalty/10
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInMasterMCTSSimulation, rewardFunction, isTerminal, rolloutHeuristic)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)


        # sample trajectory
        qPosInit = (0, ) * 24
        qVelInit = (0, ) * 24
        qPosInitNoise = 7
        qVelInitNoise = 5
        numAgent = 3
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(3, 12))
        maxRopePartLength = 0.25
        reset = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # saving trajectories
        # policy

        # wolfPolicy = HeatSeekingDiscreteDeterministicPolicy(wolfActionSpace, getWolfQPos, getSheepQPos, computeAngleBetweenVectors)
        policy = lambda state: [sheepPolicy(state[:3]), wolfPolicy(state[:3]),  mcts(state)]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()