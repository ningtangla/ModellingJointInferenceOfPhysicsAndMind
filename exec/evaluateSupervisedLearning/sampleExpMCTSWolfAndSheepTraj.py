import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, StochasticMCTS,InitializeChildren, establishPlainActionDist, \
    backup, RollOut, StochasticMCTS, establishPlainActionDistFromMultipleTrees
from src.constrainedChasingEscapingEnv.envMujoco import  ResetUniformForLeashed,TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, TransitionFunction, IsTerminal
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete, RewardFunctionWithWall,RewardFunctionAvoidCollisionAndWall
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
import mujoco_py as mujoco
import numpy as np

class IsTerminalWithRope:
    def __init__(self, minXDis, getSheepPos, getWolfPos, getRopeQPos):
        self.minXDis = minXDis
        self.getSheepPos = getSheepPos
        self.getWolfPos = getWolfPos
        self.getRopeQPos = getRopeQPos

    def __call__(self, state):
        state = np.asarray(state)
        posSheep = self.getSheepPos(state)
        posWolf = self.getWolfPos(state)
        posRope = [getPos(state) for getPos in self.getRopeQPos]
        L2NormDistanceForWolf = np.linalg.norm((posSheep - posWolf), ord=2)
        L2NormDistancesForRope = np.array([np.linalg.norm((posSheep - pos), ord=2) for pos in posRope])
        terminal = (L2NormDistanceForWolf <= self.minXDis) or np.any(L2NormDistancesForRope <= self.minXDis)

        return terminal


def main():
    # manipulated variables and other important parameters
    killzoneRadius = 1
    numSimulations = 200
    maxRunningSteps = 250
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', 'data','generateExpDemo','trajectories')

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
        preyPowerRatio = 0.7
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.07
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))


        # parametersForTrajectoryPath['preyPowerRatio'] = preyPowerRatio
        # parametersForTrajectoryPath['predatorPowerRatio'] = predatorPowerRatio
        # parametersForTrajectoryPath['masterPowerRatio'] = masterPowerRatio
        # parametersForTrajectoryPath['distractorPowerRatio'] = distractorPowerRatio
        # trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

        numActionSpace = len(actionSpace)

        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision3Agents.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        ropeIds = range(3,12)
        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
        getRopeQPos = [GetAgentPosFromState(ropeId, qPosIndex) for ropeId in ropeIds]
        isTerminalWithRope = IsTerminalWithRope(killzoneRadius, getSheepQPos, getWolfQPos, getRopeQPos)
        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)
        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

# neural network init
        numStateSpace = 18
        numActionSpace = len(actionSpace)
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        initWolfNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initMasterNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

        numStateSpaceSheep = 24
        generateSheepModel = GenerateModel(numStateSpaceSheep, numActionSpace, regularizationFactor)
        initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)


# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMasterNNModel, masterPreTrainModelPath)
        masterPolicy = ApproximatePolicy(masterPreTrainModel, masterActionSpace)

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace)

# wolf NN model
        sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        sheepPreTrainModel = restoreVariables(initSheepNNModel, sheepPreTrainModelPath)
        sheepPolicy = ApproximatePolicy(sheepPreTrainModel, sheepActionSpace)

        randomMasterPolicy = RandomPolicy(masterActionSpace)


        transitInSheepMCTSSimulation = \
                lambda state, sheepSelfAction: transit(state, [sheepSelfAction, chooseGreedyAction(wolfPolicy(state[:3])),  chooseGreedyAction(randomMasterPolicy(state))])


        transitInWolfMCTSSimulation = \
                lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(sheepPolicy(state[:4])), wolfSelfAction, chooseGreedyAction(randomMasterPolicy(state))])


# MCTS sheep
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getSheepUniformActionPrior = lambda state: {action: 1/numActionSpace for action in sheepActionSpace}
        initializeChildrenUniformPriorSheep = InitializeChildren(sheepActionSpace, transitInSheepMCTSSimulation, getSheepUniformActionPrior)
        sheepExpand = Expand(isTerminalWithRope, initializeChildrenUniformPriorSheep)

        sheepAliveBonus = 0.05
        sheepDeathPenalty = -1
        safeBound = 2
        wallDisToCenter = 10
        wallPunishRatio = 2
        velocityBound = 1
        velIndex = [4,5]
        getSheepVelocity = GetAgentPosFromState(sheepId, velIndex)
        sheepRewardFunction = RewardFunctionAvoidCollisionAndWall(sheepAliveBonus, sheepDeathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isTerminalWithRope, getSheepQPos, getSheepVelocity)

        sheepRolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
        sheepRolloutHeuristicWeight = sheepDeathPenalty/10
        sheepMaxRolloutSteps = 10
        sheepRolloutHeuristic = HeuristicDistanceToTarget(sheepRolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        sheepRollout = RollOut(sheepRolloutPolicy, sheepMaxRolloutSteps, transitInSheepMCTSSimulation, sheepRewardFunction, isTerminalWithRope,
                          sheepRolloutHeuristic)
        numTrees = 2
        numSimulationsPerTree = 100
        mctsSheep = StochasticMCTS(numTrees, numSimulationsPerTree, selectChild, sheepExpand, sheepRollout, backup, establishPlainActionDistFromMultipleTrees)
# MCTS wolf
        getWolfUniformActionPrior = lambda state: {action: 1/numActionSpace for action in wolfActionSpace}
        initializeChildrenUniformPriorWolf = InitializeChildren(wolfActionSpace, transitInWolfMCTSSimulation, getWolfUniformActionPrior)
        wolfExpand = Expand(isTerminal, initializeChildrenUniformPriorWolf)

        wolfAliveBonus = -0.05
        wolfDeathPenalty = 1
        wolfRewardFunction = RewardFunctionCompete(wolfAliveBonus, wolfDeathPenalty, isTerminal)

        wolfRolloutPolicy = lambda state: wolfActionSpace[np.random.choice(range(numActionSpace))]
        wolfRolloutHeuristicWeight = wolfDeathPenalty/10
        wolfMaxRolloutSteps = 10
        wolfRolloutHeuristic = HeuristicDistanceToTarget(wolfRolloutHeuristicWeight, getWolfQPos, getWolfQPos)
        wolfRollout = RollOut(wolfRolloutPolicy, wolfMaxRolloutSteps, transitInWolfMCTSSimulation, wolfRewardFunction, isTerminal,
                          wolfRolloutHeuristic)

        numSimulationsPerTreeForWolf = 100
        mctsWolf = StochasticMCTS(numTrees, numSimulationsPerTreeForWolf, selectChild, wolfExpand, wolfRollout, backup, establishPlainActionDistFromMultipleTrees)


        # sample trajectory
        qPosInit = (0, ) * 24
        qVelInit = (0, ) * 24
        qPosInitNoise = 6
        qVelInitNoise = 6
        numAgent = 3
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(3, 12))
        maxRopePartLength = 0.35
        reset = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)


        # saving trajectories
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # policy
        policy = lambda state: [mctsSheep(state), mctsWolf(state),  masterPolicy(state[:3])]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
