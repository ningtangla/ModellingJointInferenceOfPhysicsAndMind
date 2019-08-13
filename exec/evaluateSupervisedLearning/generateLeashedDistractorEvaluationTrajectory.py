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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,IsCollided

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

class RewardFunctionForDistractor():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isTerminal, getPosition, getVelocity):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.velocityBound = velocityBound
        self.isTerminal = isTerminal
        self.getPosition = getPosition
        self.getVelocity = getVelocity

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish =  - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0,self.safeBound -  minDisToWall), 2) / np.power(self.safeBound, 2)

        agentVel = self.getVelocity(state)
        velPunish = -np.abs(self.aliveBonus) if np.linalg.norm(agentVel) <= self.velocityBound else 0

        return reward + wallPunish + velPunish

def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning',
                                       'evaluateLeashedDistractorTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)


    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 25
    trainNumSimulations = 200
    killzoneRadius = 1
    distractorId = 3

    trajectoryFixedParameters = {'agentId': distractorId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # Mujoco environment
        physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'expLeashed.xml')
        physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
        physicsSimulation = mujoco.MjSim(physicsModel)
        numAgents = 2
        agentIds = list(range(numAgents))

        sheepId = 0
        wolfId = 1
        masterId = 2
        distractorId = 3
        qPosIndex = [0, 1]
        getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
        getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
        getMasterQPos = GetAgentPosFromState(masterId, qPosIndex)
        getDistractorQPos = GetAgentPosFromState(distractorId, qPosIndex)

        isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)


        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preyPowerRatio = 1
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 1.3
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        masterPowerRatio = 0.4
        masterActionSpace = list(map(tuple, np.array(actionSpace) * masterPowerRatio))
        distractorActionSpace = sheepActionSpace

        numActionSpace = len(wolfActionSpace)


        aliveBonus = 0.05
        deathPenalty = -1
        # rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        safeBound = 2.5
        wallDisToCenter = 10
        wallPunishRatio = 4

        velIndex = [4, 5]
        getDistractorVel =  GetAgentPosFromState(distractorId, velIndex)
        velocityBound = 5

        otherIds = list(range(13))
        otherIds.remove(distractorId)
        getOthersPos = [GetAgentPosFromState(otherId, qPosIndex) for otherId in otherIds]

        isCollided = IsCollided(killzoneRadius, getDistractorQPos, getOthersPos)
        rewardFunction = RewardFunctionForDistractor(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, velocityBound, isCollided, getDistractorQPos, getDistractorVel)


        # neural network init and save path
        numStateSpace = 24
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)


        NNFixedParameters = {'agentId': distractorId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}

        dirName = os.path.dirname(__file__)
        NNModelSaveDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning', 'leashedDistractorAvoidRopeNNModels')
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

        evalNumTrials = 500
        getResetFromTrial = lambda trial: ResetUniformForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId, \
                ropeParaIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

        evalMaxRunningSteps = 25
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal, getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]


        generateSheepModel = GenerateModel(18, numActionSpace, regularizationFactor)
        generateWolfModel = GenerateModel(18, numActionSpace, regularizationFactor)
        generateMasterModel = GenerateModel(18, numActionSpace, regularizationFactor)

        initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initWolfNNModel = generateWolfModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
        initMasterNNModel = generateMasterModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)

# sheep NN model
        sheepPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'sheepAvoidRopeModel','agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        sheepPreTrainModel = restoreVariables(initSheepNNModel, sheepPreTrainModelPath)
        sheepPolicy = ApproximatePolicy(sheepPreTrainModel, sheepActionSpace)

# wolf NN model
        wolfPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedWolfNNModels','agentId=1_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        wolfPreTrainModel = restoreVariables(initWolfNNModel, wolfPreTrainModelPath)
        wolfPolicy = ApproximatePolicy(wolfPreTrainModel, wolfActionSpace)

# master NN model
        masterPreTrainModelPath = os.path.join('..', '..', 'data', 'evaluateSupervisedLearning', 'leashedMasterNNModels','agentId=2_depth=4_learningRate=0.0001_maxRunningSteps=25_miniBatchSize=256_numSimulations=200_trainSteps=20000')
        masterPreTrainModel = restoreVariables(initMasterNNModel, masterPreTrainModelPath)
        masterPolicy = ApproximatePolicy(masterPreTrainModel, masterActionSpace)


        # save evaluation trajectories
        manipulatedVariables = json.loads(sys.argv[1])
        modelPath = getNNModelSavePath(manipulatedVariables)
        restoredModel = restoreVariables(initNNModel, modelPath)
        distractorPolicy = ApproximatePolicy(restoredModel, distractorActionSpace)

#policy
        policy = lambda state: [sheepPolicy(state[:3]), wolfPolicy(state[:3]),  masterPolicy(state[:3]), distractorPolicy(state[:4])]

        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories[startSampleIndex:endSampleIndex]]

        saveToPickle(trajectories, trajectorySavePath)
        restoredModel.close()

if __name__ == '__main__':
    main()
