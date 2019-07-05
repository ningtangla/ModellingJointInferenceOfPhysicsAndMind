import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import json
import numpy as np
import pickle
import time
import mujoco_py as mujoco

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, RollOut, Expand, \
    MCTS, backup, establishPlainActionDist
from src.episode import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.evaluationFunctions import GetSavePath
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateActionPrior, ApproximateValueFunction, \
    Train, saveVariables, restoreVariables, sampleData

def saveData(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0
    reset = Reset(physicsSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    wolfActionInSheepMCTSSimulation = lambda state: (0, 0)
    transitInSheepMCTSSimulation = lambda state, sheepSelfAction: transit(state, [sheepSelfAction, wolfActionInSheepMCTSSimulation(state)])

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    initializedNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    
    maxRunningSteps = 10
    numSimulations = 50
    NNFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations}
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)
    
    parametersForNNPath = json.loads(sys.argv[1])
    NNSavePath = getNNModelSavePath(parametersForNNPath)  
    trainedNNModel = restoreVariables(initializedNNModel, NNSavePath)

    approximateActionPrior = ApproximateActionPrior(trainedNNModel, actionSpace)
    initializeChildrenNNPrior = InitializeChildren(actionSpace, transitInSheepMCTSSimulation, approximateActionPrior)
    expandNNPrior = Expand(isTerminal, initializeChildrenNNPrior)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRolloutSteps = 5
    aliveBonus = -1/maxRolloutSteps
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)
    rolloutHeuristicWeight = 0
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepMCTSSimulation, rewardFunction, isTerminal, rolloutHeuristic)

    mctsNNPriorRolloutValue = MCTS(numSimulations, selectChild, expandNNPrior, rollout, backup, establishPlainActionDist)
    sheepPolicy = lambda state: mctsNNPriorRolloutValue(state)
    wolfPolicy = lambda state: stationaryAgentPolicy(state)
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    # sampleTrajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)
 
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                           'trajectories')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)
     
    beginTime = time.time() 
    parametersForTrajectoryPath = json.loads(sys.argv[1])
    sampleIndex = int(sys.argv[2])
    parametersForTrajectoryPath['sampleIndex'] = sampleIndex
    trajectorySavePath = getTrajectorySavePath(parametersForTrajectoryPath) 
     
    if not os.path.isfile(trajectorySavePath):
        trajectory = sampleTrajectory(policy)
        saveData(trajectory, trajectorySavePath)
    processTime = time.time() - beginTime
    print(processTime)

if __name__ == "__main__":
    main()
