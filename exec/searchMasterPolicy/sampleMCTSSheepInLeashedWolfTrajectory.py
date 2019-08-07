import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import json
from collections import OrderedDict
import pickle
import pandas as pd
import time
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniformForLeashed, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, StochasticMCTS, backup, \
    establishPlainActionDistFromMultipleTrees, RollOut
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy, RandomPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionWithWall, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards


def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'searchMasterPolicy',
                                       'mctsSheep')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'

    evalMaxRunningSteps = 7
    evalNumTrials = 2

    wolfId = 1
    trajectoryFixedParameters = {}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'leased.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)

        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        killzoneRadius = 1
        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        originActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preyPowerRatio = 0.7
        sheepActionSpace = list(map(tuple, np.array(originActionSpace) * preyPowerRatio))
        predatorPowerRatio = float(parametersForTrajectoryPath['predatorPowerRatio'])
        wolfActionSpace = list(map(tuple, np.array(originActionSpace) * predatorPowerRatio))
        masterPowerRatio = float(parametersForTrajectoryPath['masterPowerRatio'])
        masterActionSpace = list(map(tuple, np.array(originActionSpace) * masterPowerRatio))
        numActionSpace = len(originActionSpace)

        # neural network init and save path
        numStateSpace = 18
        numSheepStateSpace = 12
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        generateSheepModel = GenerateModel(numSheepStateSpace, numActionSpace, regularizationFactor)

        initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initWolfNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

        # generate a set of starting conditions to maintain consistency across all the conditions
        # sample trajectory
        qPosInit = (0, ) * 24
        qVelInit = (0, ) * 24
        qPosInitNoise = 7
        qVelInitNoise = 0
        numAgent = 3
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(3, 12))
        maxRopePartLength = 0.25

        getResetFromTrial = lambda trial: ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

        # save evaluation trajectories
        wolfModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        restoredWolfModel = restoreVariables(initWolfNNModel, wolfModelPath)
        wolfPolicy = ApproximatePolicy(restoredWolfModel, wolfActionSpace)

        masterPolicy = RandomPolicy(masterActionSpace)

        transitInSheepMCTSSimulation = \
                lambda state, sheepSelfAction: transit(state, [sheepSelfAction, chooseGreedyAction(wolfPolicy(state[0:3])), \
                chooseGreedyAction(masterPolicy(state))])
        # MCTS
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
        safeBound = 1.5
        wallDisToCenter = 10
        wallPunishRatio = 1.5
        rewardFunction = RewardFunctionWithWall(aliveBonus, deathPenalty,safeBound, wallDisToCenter, wallPunishRatio, isTerminal, getSheepXPos)

        rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeightToOtherAgent = -0.1
        maxRolloutSteps = 7
        rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeightToOtherAgent,  getWolfXPos, getSheepXPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        numTrees = 4
        numSimulationsPerTree = int(parametersForTrajectoryPath['numSimulationsPerTree'])
        mcts = StochasticMCTS(numTrees, numSimulationsPerTree, selectChild, expand, rollout, backup, establishPlainActionDistFromMultipleTrees)


#policy
        policy = lambda state: [mcts(state), wolfPolicy(state[:3]), stationaryAgentPolicy(state)]

        beginTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        processTime = time.time() - beginTime
        saveToPickle(trajectories, trajectorySavePath)
        restoredWolfModel.close()

if __name__ == '__main__':
    main()
