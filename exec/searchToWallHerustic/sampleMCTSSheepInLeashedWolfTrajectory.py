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
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist, RollOut
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingDiscreteDeterministicPolicy, \
    HeatSeekingContinuesDeterministicPolicy
from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, GenerateAllSampleIndexSavePaths, SaveAllTrajectories, saveToPickle
from exec.evaluationFunctions import ComputeStatistics, GenerateInitQPosUniform
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget, HeuristicDistanceToOtherAgentAndWall
from exec.preProcessing import AccumulateRewards


def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'searchToWallHerustic',
                                       'mctsSheep')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 25
    trainNumSimulations = 200
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
        preyPowerRatio = float(parametersForTrajectoryPath['preyPowerRatio'])
        sheepActionSpace = list(map(tuple, np.array(originActionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        actionSpace = list(map(tuple, np.array(originActionSpace) * predatorPowerRatio))

        numActionSpace = len(actionSpace)

        # neural network init and save path
        numStateSpace = 18
        numSheepStateSpace = 12
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
        generateSheepModel = GenerateModel(numSheepStateSpace, numActionSpace, regularizationFactor)

        initNNModel = generateModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

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

        evalNumTrials = 3
        getResetFromTrial = lambda trial: ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
        evalMaxRunningSteps = 50
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

        # save evaluation trajectories
        modelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')

        restoredModel = restoreVariables(initNNModel, modelPath)
        wolfPolicy = ApproximatePolicy(restoredModel, actionSpace)


        transitInSheepMCTSSimulation = \
                lambda state, sheepSelfAction: transit(state, [sheepSelfAction, chooseGreedyAction(wolfPolicy(state[0:3])), \
                chooseGreedyAction(stationaryAgentPolicy(state))])
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
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        rolloutPolicy = lambda state: sheepActionSpace[np.random.choice(range(numActionSpace))]
        rolloutHeuristicWeightToOtherAgent = -0.1
        rolloutHeuristicWeightToWall = float(parametersForTrajectoryPath['heuristicWeightWallDis'])
        maxRolloutSteps = 7
        wallDisToCenter = 10
        rolloutHeuristic = HeuristicDistanceToOtherAgentAndWall(rolloutHeuristicWeightToOtherAgent, rolloutHeuristicWeightToWall, wallDisToCenter, getWolfXPos, getSheepXPos)
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal,
                          rolloutHeuristic)

        numSimulations = 100
        mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)


#policy
        policy = lambda state: [mcts(state), wolfPolicy(state[:3]), stationaryAgentPolicy(state)]

        beginTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories]
        processTime = time.time() - beginTime
        saveToPickle(trajectories, trajectorySavePath)
        restoredModel.close()

if __name__ == '__main__':
    main()
