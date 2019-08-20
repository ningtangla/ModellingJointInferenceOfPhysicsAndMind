import sys
import os
import json
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut, StochasticMCTS, establishPlainActionDistFromMultipleTrees
from src.constrainedChasingEscapingEnv.envMujoco import  ResetUniformForLeashed,TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, TransitionFunction
from src.episode import SampleTrajectory, chooseGreedyAction, sampleAction
from exec.trajectoriesSaveLoad import GetSavePath, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete, RewardFunctionWithWall
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from exec.trajectoriesSaveLoad import readParametersFromDf
from exec.parallelComputing import GenerateTrajectoriesParallel
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
import mujoco_py as mujoco
import numpy as np

class IsTerminal:
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
    numSimulations = 150
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
        predatorPowerRatio = 1.4
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.1
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorPowerRatio = 0.8
        distractorActionSpace = list(map(tuple, np.array(actionSpace) * distractorPowerRatio))

        # parametersForTrajectoryPath['preyPowerRatio'] = preyPowerRatio
        # parametersForTrajectoryPath['predatorPowerRatio'] = predatorPowerRatio
        # parametersForTrajectoryPath['masterPowerRatio'] = masterPowerRatio
        # parametersForTrajectoryPath['distractorPowerRatio'] = distractorPowerRatio
        # trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

        numActionSpace = len(actionSpace)

        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        ropeIds = range(4,13)
        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
        getRopeQPos = [GetAgentPosFromState(ropeId, qPosIndex) for ropeId in ropeIds]
        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos, getRopeQPos)

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

        distractorNumStateSpace = 24
        generateDistractorModel = GenerateModel(distractorNumStateSpace, numActionSpace, regularizationFactor)
        initdistractorNNModel = generateDistractorModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMasterNNModel, masterPreTrainModelPath)
        masterPolicy = ApproximatePolicy(masterPreTrainModel, masterActionSpace)

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace)

# distractor NN model
        distractorPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorAvoidRopeNNModels','agentId=3_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=100000')
        distractorPreTrainModel = restoreVariables(initdistractorNNModel, distractorPreTrainModelPath)
        distractorPolicy = ApproximatePolicy(distractorPreTrainModel, distractorActionSpace)



        transitInSheepMCTSSimulation = \
                lambda state, sheepSelfAction: transit(state, [sheepSelfAction, chooseGreedyAction(wolfPolicy(state[:3])),  chooseGreedyAction(masterPolicy(state[0:3])),
                chooseGreedyAction(distractorPolicy(state[:4]))])


# MCTS sheep
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in sheepActionSpace}
        initializeChildrenUniformPrior = InitializeChildren(sheepActionSpace, transitInSheepMCTSSimulation,
                                                            getUniformActionPrior)
        expand = Expand(isTerminal, initializeChildrenUniformPrior)

        aliveBonus = 0.05
        deathPenalty = -1
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)
        safeBound = 1.5
        wallDisToCenter = 10
        wallPunishRatio = 1.5
        # rewardFunction = RewardFunctionWithWall(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal, getSheepQPos)

        rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeight = deathPenalty/10
        maxRolloutSteps = 10
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfQPos, getSheepQPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        numTrees = 2
        numSimulationsPerTree = 100
        # mcts = StochasticMCTS(numTrees, numSimulationsPerTree, selectChild, expand, rollout, backup, establishPlainActionDistFromMultipleTrees)

        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

        # sample trajectory
        qPosInit = (0, ) * 26
        qVelInit = (0, ) * 26
        qPosInitNoise = 6
        qVelInitNoise = 5
        numAgent = 4
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(4, 13))
        maxRopePartLength = 0.35
        reset = ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

        # saving trajectories
        # policy

        policy = lambda state: [mcts(state), wolfPolicy(state[:3]),  masterPolicy(state[:3]), distractorPolicy(state[:4])]

        # generate trajectories
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()