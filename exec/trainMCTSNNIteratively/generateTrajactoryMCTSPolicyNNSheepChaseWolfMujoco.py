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
from src.play import SampleTrajectory, agentDistToGreedyAction, worldDistToAction
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

    # functions for MCTS
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
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentPolicy(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionPriorUniform = {action: 1/len(actionSpace) for action in actionSpace}
    getActionPriorUniform = lambda state: actionPriorUniform

    initializeChildren = InitializeChildren(actionSpace, sheepTransit, getActionPriorUniform)
    expand = Expand(isTerminal, initializeChildren)

    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfXPos, getSheepXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # All agents' policies
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)
    random = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    sheepPolicies = {'mcts': mcts, 'random': random}
    condition = json.loads(sys.argv[1])
    sheepPolicyName = condition['sheepPolicyName']
    wolfPolicy = lambda state: stationaryAgentPolicy(state)
    sheepPolicy = lambda state: sheepPolicies[sheepPolicyName](state)
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    # sampleTrajectory
    distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, distToAction)
 
    # savePath
    saveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainNNIteratively',
                                      'trajectory')

    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    extension = '.pickle'
    getSavePath = GetSavePath(saveDirectory, extension)
     
    beginTime = time.time()
    
    parametersForPath = condition
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
