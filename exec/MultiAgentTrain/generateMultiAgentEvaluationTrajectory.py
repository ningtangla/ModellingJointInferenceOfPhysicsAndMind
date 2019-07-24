import sys
import os
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

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
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


class RestoreNNModel:
    def __init__(self, getModelSavePath, multiAgentNNModel, restoreVariables):
        self.getModelSavePath = getModelSavePath
        self.multiAgentNNModel = multiAgentNNModel
        self.restoreVariables = restoreVariables

    def __call__(self, agentId, iteration):
        modelPath = self.getModelSavePath({'agentId': agenId, 'iterationIndex': iteration})
        restoredNNModel = self.restoreVariables(self.multiAgentNNModel[agentId], modelPath)

        return restoredNNModel


class PreparePolicy:
    def __init__(self, selfApproximatePolicy, otherApproximatePolicy):
        self.selfApproximatePolicy = selfApproximatePolicy
        self.otherApproximatePolicy = otherApproximatePolicy

    def __call__(self, agentId, multiAgentNNModel):
        multiAgentPolicy = [self.otherApproximatePolicy(NNModel) for NNModel in multiAgentNNModel]
        selfNNModel = multiAgentNNModel[agentId]
        multiAgentPolicy[agentId] = self.selfApproximatePolicy(selfNNModel)
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy

def main():
    # Mujoco environment
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    numAgents = 2
    agentIds = list(range(numAgents))

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)
    
    # neural network init and save path
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    trainMaxRunningSteps = 20
    trainNumSimulations = 20

    evalNumSimulations = 20  # 200
    evalNumTrials = 1000
    evalMaxRunningSteps = 20
    NNFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}
    dirName = os.path.dirname(__file__)
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgent', 'NNModel')
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

    # functions to get prediction from NN
    restoreNNModel = RestoreNNModel(getNNModelSavePath, initializedNNModel, restoreVariables)
    NNPolicy = ApproximatePolicy(initializedNNModel, actionSpace)

    # policy

    # generate a set of starting conditions to maintain consistency across all the conditions
    evalQPosInitNoise = 0
    evalQVelInitNoise = 0
    qVelInit = [0, 0, 0, 0]

    getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents,
                                                              evalQPosInitNoise, evalQVelInitNoise)

    generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
    evalAllQPosInit = [generateInitQPos() for _ in range(evalNumTrials)]
    evalAllQVelInit = np.random.uniform(-8, 8, (evalNumTrials, 4))
    getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, evalAllQPosInit[trial], evalAllQVelInit[trial],
                                                   numAgents, evalQPosInitNoise, evalQVelInitNoise)
    getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal,
                                                         getResetFromTrial(trial), chooseGreedyAction)

    allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

    # save evaluation trajectories
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data',
                                        'multiAgentTrain', 'multiMCTSAgent', 'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    parametersForTrajectoryPath = json.loads(sys.argv[1])
    startSampleIndex = int(sys.argv[2])
    endSampleIndex = int(sys.argv[3])

    selfIteration = parametersForTrajectoryPath['selfIteration']
    selfId = parametersForTrajectoryPath['agentId']
    otherIds = [ :selfId] + [selfId+1: ] 
    restoreNNModel(iteration, agentId)
    policy = preparePolicy(policyName)
    parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath)
    #if not os.path.isfile(trajectorySavePath):
    #parametersForTrajectoryPath['sampleTrajectoryTime'] = processTime
    beginTime = time.time()
    trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories[startSampleIndex:endSampleIndex]]
    processTime = time.time() - beginTime
    saveToPickle(trajectories, trajectorySavePath)
    
if __name__ == '__main__':
    main()
