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
from src.play import SampleTrajectory, chooseGreedyAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState
from exec.evaluationFunctions import GetSavePath
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy

def saveData(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


def main():
    maxRunningSteps = 20
    qPosInit = (0, 0, 0, 0)
    numSimulations = 75

    dirName = os.path.dirname(__file__)
    evalEnvModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    evalModel = mujoco.load_model_from_path(evalEnvModelPath)
    evalSimulation = mujoco.MjSim(evalModel)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0
    reset = Reset(evalSimulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    renderOn = False
    numSimulationFrames = 20
    transit = TransitionFunction(evalSimulation, isTerminal, numSimulationFrames)
    wolfPolicyInSheepSimulation = lambda state: stationaryAgentPolicy(state)
    policyInSheepSimulation = lambda state, sheepSelfAction: [sheepSelfAction, wolfPolicyInSheepSimulation(state)])
    transitInSheepSimulation = lambda state, sheepSelfAction: transit(policyInSheepSimulation(state, sheepSelfAction))

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)
    
    parametersForNNPath = json.loads(sys.argv[1])
    NNModel = 
    approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    initializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, transitInSheepSimulation, approximateActionPrior(NNModel))
    expandNNPrior = lambda NNModel: Expand(isTerminal, initializeChildrenNNPrior(NNModel))

    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInSheepSimulation, rewardFunction, isTerminal, rolloutHeuristic)

    # All agents' policies
    mctsNNPriorRolloutValue = MCTS(numSimulations, selectChild, expandNNPrior, rollout, backup, establishPlainActionDist)
    sheepPolicy = lambda state: mctsNNPriorRolloutValue(state)
    wolfPolicy = lambda state: stationaryAgentPolicy(state)
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    # sampleTrajectory
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)
 
    # savePath
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBuffer',
                                      'trajectory')

    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)

    extension = '.pickle'
    getSavePath = GetSavePath(trajectorySaveDirectory, extension)
     
    beginTime = time.time()
    
    sampleIndex = int(sys.argv[2]) 
    parametersForPath['sampleIndex'] = sampleIndex
    trajectorySavePath = getSavePath(parametersForPath) 
     
    if not os.path.isfile(trajectorySavePath):
        trajectory = sampleTrajectory(policy)
        saveData(trajectory, trajectorySavePath)
    processTime = time.time() - beginTime
    print(processTime)

if __name__ == "__main__":
    main()
