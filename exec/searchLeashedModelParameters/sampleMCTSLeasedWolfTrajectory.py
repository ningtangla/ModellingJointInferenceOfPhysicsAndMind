import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, TransitionFunction, ResetUniformForLeashed
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximatePolicy,restoreVariables
import mujoco_py as mujoco
import numpy as np




def main():
    # manipulated variables and other important parameters
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                             'searchLeashedModelParameters', 'leasedTrajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension)


    parametersForCondition = json.loads(sys.argv[1])

    trajectorySavePath = generateTrajectorySavePath(parametersForCondition)

    if not os.path.isfile(trajectorySavePath):
        # Mujoco Environment
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'leased.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)
        numTendon = physicsSimulation.model.ntendon
        predatorBodyIndex = 7
        draggerBodyIndex = 8

        #set physical model parameter
        tendonStiffness = float(parametersForCondition['tendonStiffness'])
        tendonDamping = float(parametersForCondition['tendonDamping'])
        maxTendonLength = float(parametersForCondition['maxTendonLength'])
        predatorMass = int(parametersForCondition['predatorMass'])
        draggerMass = int(parametersForCondition['draggerMass'])
        predatorPowerRatio = float(parametersForCondition['predatorPower'])

        physicsSimulation.model.tendon_stiffness[:] = [tendonStiffness] * numTendon
        physicsSimulation.model.tendon_damping[:] = [tendonDamping] * numTendon
        physicsSimulation.model.tendon_range[:] = [[0, maxTendonLength]] * numTendon
        physicsSimulation.model.body_mass[[predatorBodyIndex, draggerBodyIndex]] = [predatorMass, draggerMass]
        physicsSimulation.set_constants()
        physicsSimulation.forward()

        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        print(wolfActionSpace)

        sheepId = 0
        wolfId = 1
        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
        killzoneRadius = 0.3
        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
        randomPolicy = RandomPolicy(actionSpace)

        # neural network init
        numStateSpace = 12
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        depth = 4
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        initModel = generateModel(sharedWidths*depth, actionLayerWidths, valueLayerWidths)

        sheepNNModelPath = os.path.join('..','..','data','searchLeashedModelParameters', 'NNModel', \
                'agentId=0_depth=4_learningRate=0.001_maxRunningSteps=25_miniBatchSize=64_numSimulations=100_trainSteps=40000')
        sheepNNModel = restoreVariables(initModel, sheepNNModelPath)
        sheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace)
        transitInWolfMCTSSimulation = \
                lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(sheepPolicy(state[0:2])), wolfSelfAction, chooseGreedyAction(randomPolicy(state))])

        # WolfActionInSheepSimulation = lambda state: (0, 0)
        # transitInSheepMCTSSimulation = \
        #     lambda state, sheepSelfAction: transit(state, [sheepSelfAction, WolfActionInSheepSimulation(state)])

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in wolfActionSpace}
        initializeChildrenUniformPrior = InitializeChildren(wolfActionSpace, transitInWolfMCTSSimulation,
                                                            getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = -0.05
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = 0.1
        maxRolloutSteps = 5
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        numSimulations = 200
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in wolfActionSpace}
        initializeChildrenUniformPrior = InitializeChildren(wolfActionSpace, transitInWolfMCTSSimulation,
                                                            getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = -0.05
        deathPenalty = 1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = 0.1
        maxRolloutSteps = 5
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        numSimulations = 200
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # sample trajectory
        qPosInit = (0, ) * 24
        qVelInit = (0, ) * 24
        qPosInitNoise = 5
        qVelInitNoise = 3
        numAgent = 3
        tiedAgentId = [1, 2]
        ropePartIndex = list(range(3, 12))
        maxRopePartLength = maxTendonLength
        reset = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        maxRunningSteps = 100
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # saving trajectories
        # policy
        policy = lambda state: [sheepPolicy(state[0:2]), mcts(state), randomPolicy(state)]

        # generate trajectories
        numTrajectories = 3
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(numTrajectories)]
        saveToPickle(trajectories, trajectorySavePath)
        print(trajectories[0][0][0])
if __name__ == '__main__':
    main()