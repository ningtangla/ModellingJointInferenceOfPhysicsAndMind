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

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniformForLeashed, TransitionFunction
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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards

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
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning',
                                       'evaluateSheepAvoidRopeTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)


    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 25
    trainNumSimulations = 200
    killzoneRadius = 1
    sheepId = 0
    trajectoryFixedParameters = {'agentId': sheepId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

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

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preyPowerRatio = 0.7
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.3
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorPowerRatio = 0.7
        distractorActionSpace = list(map(tuple, np.array(actionSpace) * distractorPowerRatio))
        numActionSpace = len(wolfActionSpace)

        alivePenalty = 0.05
        deathBonus = -1
        rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

        # neural network init and save path
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(24, numActionSpace, regularizationFactor)
        generateModelWolf = GenerateModel(18, numActionSpace, regularizationFactor)
        generateModelMaster = GenerateModel(18, numActionSpace, regularizationFactor)
        generateModelDistractor = GenerateModel(24, numActionSpace, regularizationFactor)
        initWolfNNModel = generateModelWolf(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initMasterNNModel = generateModelMaster(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initDistractorNNModel = generateModelDistractor(sharedWidths * 4, actionLayerWidths, valueLayerWidths)




        NNFixedParameters = {'agentId': sheepId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

        dirName = os.path.dirname(__file__)
        NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel')
        NNModelSaveExtension = ''
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

        depth = int(parametersForTrajectoryPath['depth'])
        initNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths)

        # generate a set of starting conditions to maintain consistency across all the conditions
        # sample trajectory
        qPosInit = (0, ) * 26
        qVelInit = (0, ) * 26
        qPosInitNoise = 6
        qVelInitNoise = 5
        numAgent = 4
        tiedAgentId = [1, 2]
        ropeParaIndex = list(range(4, 13))
        maxRopePartLength = 0.35

        evalNumTrials = 1000
        getResetFromTrial = lambda trial: ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        evalMaxRunningSteps = 25
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]


# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMasterNNModel, masterPreTrainModelPath)
        masterPolicy = ApproximatePolicy(masterPreTrainModel, masterActionSpace)

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace)

# distractor NN model
        distractorPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorNNModels','agentId=3_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        distractorPreTrainModel = restoreVariables(initDistractorNNModel, distractorPreTrainModelPath)
        distractorPolicy = ApproximatePolicy(distractorPreTrainModel, distractorActionSpace)


        # save evaluation trajectories
        manipulatedVariables = json.loads(sys.argv[1])
        modelPath = getNNModelSavePath(manipulatedVariables)
        restoredModel = restoreVariables(initNNModel, modelPath)
        sheepPolicy = ApproximatePolicy(restoredModel, sheepActionSpace)

#policy
        policy = lambda state: [sheepPolicy(state[0:4]), wolfPolicy(state[:3]), masterPolicy(state[:3]),distractorPolicy(state[:4])]

        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories[startSampleIndex:endSampleIndex]]

        saveToPickle(trajectories, trajectorySavePath)
        restoredModel.close()

if __name__ == '__main__':
    main()
