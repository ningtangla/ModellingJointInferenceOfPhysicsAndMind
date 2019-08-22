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
from src.neuralNetwork.policyValueResNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.measure import DistanceBetweenActualAndOptimalNextPosition, \
    ComputeOptimalNextPos, GetAgentPosFromTrajectory, GetStateFromTrajectory
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete, RewardFunctionWithWall, RewardFunctionAvoidCollisionAndWall, IsCollided
from exec.preProcessing import AccumulateRewards


def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateResNN',
                                       'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 25
    trainNumSimulations = 100
    killzoneRadius = 1
    wolfId = 1
    trajectoryFixedParameters = {'agentId': wolfId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'noRopeCollision.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)
        numAgents = 2
        agentIds = list(range(numAgents))

        sheepId = 0
        wolfId = 1
        xPosIndex = [2, 3]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        numActionSpace = len(actionSpace)

        alivePenalty = -0.05
        deathBonus = 1
        rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

        # neural network init and save path
        numStateSpace = 78
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)


        NNFixedParameters = {'agentId': wolfId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations,'killzoneRadius':killzoneRadius}
        dirName = os.path.dirname(__file__)
        NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateResNN',
                                            'trainedRes2WolfModels')
        NNModelSaveExtension = ''
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

        depth = int(parametersForTrajectoryPath['depth'])
        resBlock = 2
        initNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlock)

        # generate a set of starting conditions to maintain consistency across all the conditions

        qPosInit = (0, ) * 26
        qVelInit = (0, ) * 26
        qPosInitNoise = 6
        qVelInitNoise = 6
        numAgent = 4
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(4, 13))
        maxRopePartLength = 0.35

        evalNumTrials = 300
        getResetFromTrial = lambda trial: ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)


        evalMaxRunningSteps = 25
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal,
                                                             getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

        # save evaluation trajectories
        manipulatedVariables = json.loads(sys.argv[1])
        modelPath = getNNModelSavePath(manipulatedVariables)
        restoredModel = restoreVariables(initNNModel, modelPath)

        wolfPolicy = ApproximatePolicy(restoredModel, wolfActionSpace)
        sheepPolicy = stationaryAgentPolicy
        masterPolicy = stationaryAgentPolicy
        distractorPolicy = stationaryAgentPolicy
        policy = lambda state: [sheepPolicy(state[:4]), wolfPolicy(state),  masterPolicy(state[:4]), distractorPolicy(state[:4])]

        beginTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories[startSampleIndex:endSampleIndex]]
        processTime = time.time() - beginTime
        saveToPickle(trajectories, trajectorySavePath)
        restoredModel.close()

if __name__ == '__main__':
    main()
