import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import pickle
from mujoco_py import load_model_from_path, MjSim
import json

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, RollOut, Expand, MCTS, backup, \
    establishPlainActionDist
from src.play import SampleTrajectory, agentDistToGreedyAction, worldDistToAction
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNetTemp import GenerateModelSeparateLastLayer, restoreVariables, \
    ApproximateActionPrior, ApproximateValueFunction                                                                    # need to remove this file
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState


def main():
    # manipulated variables (and some other parameters that are commonly varied)
    maxRunningSteps = 2
    trialConditionParameters = json.loads(sys.argv[1])
    trainSteps = trialConditionParameters['trainSteps']
    sheepPolicyName = trialConditionParameters['sheepPolicyName']
    numSimulations = trialConditionParameters['numSimulations']
    sampleIndex = sys.argv[2]

    # Mujoco environment
    dirName = os.path.dirname(__file__)
    mujocoModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    mujocoModel = load_model_from_path(mujocoModelPath)
    simulation = MjSim(mujocoModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 1
    reset = Reset(simulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)

    killzoneRadius = 0.5
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transit = TransitionFunction(simulation, isTerminal, numSimulationFrames)
    stationaryAgentAction = lambda state: agentDistToGreedyAction(stationaryAgentPolicy(state))
    sheepTransit = lambda state, action: transit(state, [action, stationaryAgentAction(state)])

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    # functions for MCTS
    cInit = 1
    cBase = 100
    scoreChild = ScoreChild(cInit, cBase)
    selectChild = SelectChild(scoreChild)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

    rolloutHeuristicWeight = 0
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getSheepXPos, getWolfXPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    # load trained neural network model
    numStateSpace = 12
    learningRate = 0.0001
    regularizationFactor = 1e-4
    hiddenWidths = [128, 128]
    generatePolicyNet = GenerateModelSeparateLastLayer(numStateSpace, numActionSpace, learningRate, regularizationFactor)

    dataSetMaxRunningSteps = 10
    dataSetNumSimulations = 75
    dataSetNumTrials = 1500
    dataSetQPosInit = (0, 0, 0, 0)
    modelTrainFixedParameters = {'maxRunningSteps': dataSetMaxRunningSteps, 'qPosInit': dataSetQPosInit,
                                 'numSimulations': dataSetNumSimulations, 'numTrials': dataSetNumTrials,
                                 'learnRate': learningRate, 'output': 'policyValue'}
    modelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco',
                                      'trainedModels')
    if not os.path.exists(modelSaveDirectory):
        os.makedirs(modelSaveDirectory)
    modelSaveExtension = ''

    getModelSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, modelTrainFixedParameters)
    modelSavePath = getModelSavePath({'trainSteps': trainSteps})
    trainedModel = restoreVariables(generatePolicyNet(hiddenWidths), modelSavePath)

    # wrapper function for expand
    uniformActionPrior = lambda state: {action: 1/len(actionSpace) for action in actionSpace}
    expandUniformPrior = Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit, uniformActionPrior))
    expandNNPrior = Expand(isTerminal, InitializeChildren(actionSpace, sheepTransit,
                                                             ApproximateActionPrior(trainedModel, actionSpace)))

    # wrapper function for approximate value function of NN
    getStateFromNode = lambda node: list(node.id.values())[0]
    approximateValueFunction = ApproximateValueFunction(trainedModel)
    getNNValue = lambda node: (approximateValueFunction(getStateFromNode(node)))

    # prepare sheepPolicy
    mcts = MCTS(numSimulations, selectChild, expandUniformPrior, rollout, backup, establishPlainActionDist)
    mctsNN = MCTS(numSimulations, selectChild, expandNNPrior, getNNValue, backup, establishPlainActionDist)
    sheepPolicies = {'MCTS': mcts, 'MCTSNN': mctsNN}
    sheepPolicy = sheepPolicies[sheepPolicyName]

    # all agents' policies
    policy = lambda state: [sheepPolicy(state), stationaryAgentPolicy(state)]

    # sample trajectory
    distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, distToAction)

    # path to save evaluation trajectories
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'testMCTSvsMCTSNNPolicyValueSheepChaseWolfMujoco',
                                       'parallel', 'evalTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    extension = '.pickle'
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit, 'qPosInitNoise': qPosInitNoise}
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, extension, trajectoryFixedParameters)
    trajectorySavePath = getTrajectorySavePath({'trainSteps': trainSteps, 'numSimulations': numSimulations,
                                                'sheepPolicyName': sheepPolicyName, 'sampleIndex': sampleIndex})

    if not os.path.isfile(trajectorySavePath):
        trajectory = sampleTrajectory(policy)
        pickleOut = open(trajectorySavePath, 'wb')
        pickle.dump(trajectory, pickleOut)
        pickleOut.close()


if __name__ == '__main__':
    main()