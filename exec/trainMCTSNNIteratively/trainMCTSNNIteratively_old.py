import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import random
import pickle
from mujoco_py import load_model_from_path, MjSim

from src.constrainedChasingEscapingEnv.envMujoco import Reset, IsTerminal, TransitionFunction
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, \
    establishPlainActionDist, RollOut
from src.play import SampleTrajectory
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, RandomPolicy
from exec.evaluationFunctions import GetSavePath
from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateActionPrior, ApproximateValueFunction, \
    Train, saveVariables, sampleData, evaluate
from src.constrainedChasingEscapingEnv.wrappers import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter
from exec.trainMCTSNNIteratively.wrappers import GetApproximateValueFromNode, getStateFromNode
from src.play import agentDistToGreedyAction, worldDistToAction
from exec.preProcessing import AddValuesToTrajectory, AccumulateRewards


def saveData(data, path):
    pickleOut = open(path, 'wb')
    pickle.dump(data, pickleOut)
    pickleOut.close()


class ActionToOneHot:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, action):
        oneHotAction = np.asarray([1 if (np.array(action) == np.array(self.actionSpace[index])).all() else 0 for index
                                   in range(len(self.actionSpace))])
        return oneHotAction


class ConstantLearningRateModifier:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def __call__(self, trainIteration):
        return self.learningRate


class PreProcessTrajectories:
    def __init__(self, agentId, actionIndex, actionToOneHot, addValuesToTrajectory):
        self.agentId = agentId
        self.actionIndex = actionIndex
        self.actionToOneHot = actionToOneHot
        self.addValuesToTrajectory = addValuesToTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addValuesToTrajectory(trajectory) for trajectory in trajectories]
        stateActionValueTriples = [triple for trajectory in trajectoriesWithValues for triple in trajectory]
        triplesFiltered = list(filter(lambda triple: triple[self.actionIndex] is not None, stateActionValueTriples))
        processTriple = lambda state, actions, actionDist, value: (np.asarray(state).flatten(), self.actionToOneHot(actions[self.agentId]), value)
        triplesProcessed = [processTriple(state, actions, actionDist, value) for state, actions, actionDist, value in triplesFiltered]

        random.shuffle(triplesProcessed)
        trainData = [[state for state, action, value in triplesProcessed],
                     [action for state, action, value in triplesProcessed],
                     np.asarray([value for state, action, value in triplesProcessed])]

        return trainData


class GetPolicy:
    def __init__(self, getSheepPolicy, getWolfPolicy):
        self.getSheepPolicy = getSheepPolicy
        self.getWolfPolicy = getWolfPolicy

    def __call__(self, NNModel):
        sheepPolicy = self.getSheepPolicy(NNModel)
        wolfPolicy = self.getWolfPolicy(NNModel)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        return policy


class PlayToTrain:
    def __init__(self, numTrialsEachIteration, sampleTrajectory, getTrajectorySavePath, getPolicy, saveData):
        self.numTrialsEachIteration = numTrialsEachIteration
        self.sampleTrajectory = sampleTrajectory
        self.getTrajectorySavePath = getTrajectorySavePath
        self.getPolicy = getPolicy
        self.saveData = saveData

    def __call__(self, NNModel, pathParameters):
        policy = self.getPolicy(NNModel)
        trajectories = [self.sampleTrajectory(policy) for trial in range(self.numTrialsEachIteration)]
        trajectorySavePath = self.getTrajectorySavePath(pathParameters)
        self.saveData(trajectories, trajectorySavePath)

        return trajectories


class TrainToPlay:
    def __init__(self, train, getModelSavePath):
        self.train = train
        self.getModelSavePath = getModelSavePath

    def __call__(self, NNModel, trainData, pathParameters):
        updatedNNModel = self.train(NNModel, trainData)
        modelSavePath = self.getModelSavePath(pathParameters)
        saveVariables(updatedNNModel, modelSavePath)

        return updatedNNModel


def main():
    # commonly varied parameters
    numIterations = 25 #100
    numTrialsEachIteration = 1

    # functions for MCTS
    dirName = os.path.dirname(__file__)
    mujocoModelPath = os.path.join(dirName, '..', '..', 'env', 'xmls', 'twoAgents.xml')
    mujocoModel = load_model_from_path(mujocoModelPath)
    simulation = MjSim(mujocoModel)
    qPosInit = (0, 0, 0, 0)
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 9.7
    qVelInitNoise = 0

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

    cInit = 1
    cBase = 100
    scoreChild = ScoreChild(cInit, cBase)
    selectChild = SelectChild(scoreChild)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)

    maxRunningSteps = 10
    numSimulations = 75                                                                                                 # should we change this number?

    # wrapper function for expand
    approximateActionPrior = lambda NNModel: ApproximateActionPrior(NNModel, actionSpace)
    getInitializeChildrenNNPrior = lambda NNModel: InitializeChildren(actionSpace, sheepTransit, approximateActionPrior(NNModel))
    getExpandNNPrior = lambda NNModel: Expand(isTerminal, getInitializeChildrenNNPrior(NNModel))

    # wrapper function for policy
    getApproximateValue = lambda NNModel: GetApproximateValueFromNode(getStateFromNode, ApproximateValueFunction(NNModel))
    getMCTSNN = lambda NNModel: MCTS(numSimulations, selectChild, getExpandNNPrior(NNModel),
                                     getApproximateValue(NNModel), backup, establishPlainActionDist)
    getStationaryAgentPolicy = lambda NNModel: stationaryAgentPolicy                                                    # should I do this just to keep the interface symmetric?
    getPolicy = GetPolicy(getMCTSNN, getStationaryAgentPolicy)

    # sample trajectory
    reset = Reset(simulation, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    distToAction = lambda worldDist: worldDistToAction(agentDistToGreedyAction, worldDist)
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, distToAction)

    # pre-process the trajectory for training the neural network
    playAliveBonus = -0.05
    playDeathPenalty = 1
    playKillzoneRadius = 0.5
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepXPos, getWolfXPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(actionSpace)
    preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot, addValuesToTrajectory)

    # NN Model
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    getModel = lambda: generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # function to train NN model
    learningRate = 1e-6
    trainStepsEachIteration = 1
    batchSizeForTrainFunction = 0
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    afterActionCoeff = 1
    afterValueCoeff = 1
    initCoeffs = (initActionCoeff, initValueCoeff)
    afterCoeffs = (afterActionCoeff, afterValueCoeff)
    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientCotroller(initCoeffs, afterCoeffs)
    reportInterval = 25
    trainReporter = TrainReporter(trainStepsEachIteration, reportInterval)
    train = Train(trainStepsEachIteration, batchSizeForTrainFunction, sampleData,
                  ConstantLearningRateModifier(learningRate), terminalController, coefficientController,
                  trainReporter)

    # NN model save path
    trainFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInitNoise': qPosInitNoise, 'qPosInit': qPosInit,
                            'numSimulations': numSimulations, 'numTrialsEachIteration': numTrialsEachIteration}
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'noReplayBuffer',
                                        'trainedNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, trainFixedParameters)

    # trajectory save path
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'qPosInit': qPosInit,
                                 'qPosInitNoise': qPosInitNoise, 'numSimulations': numSimulations,
                                 'numTrialsEachIteration': numTrialsEachIteration}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'noReplayBuffer',
                                        'trajectories', 'train')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    # play and train NN iteratively
    playToTrain = PlayToTrain(numTrialsEachIteration, sampleTrajectory, getTrajectorySavePath, getPolicy, saveData)
    trainToPlay = TrainToPlay(train, getModelSavePath)

    # validation data
    validationSize = 20
    validationNumSimulation = 75
    validationMaxRunningSteps = 10
    uniformActionPrior = lambda state: {action: 1/len(actionSpace) for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, sheepTransit, uniformActionPrior)
    expandUniformPrior = Expand(isTerminal, initializeChildrenUniformPrior)
    randomPolicy = RandomPolicy(actionSpace)
    rolloutPolicy = lambda state: agentDistToGreedyAction(randomPolicy(state))
    maxRollOutSteps = 10
    validationAliveBonus = -0.05
    validationDeathPenalty = 1
    rewardFunction = RewardFunctionCompete(validationAliveBonus, validationDeathPenalty, isTerminal)
    rolloutHeuristicWeight = 0
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getSheepXPos, getWolfXPos)
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, sheepTransit, rewardFunction, isTerminal, rolloutHeuristic)

    mcts = MCTS(validationNumSimulation, selectChild, expandUniformPrior, rollout, backup, establishPlainActionDist)
    validationPolicy = lambda state: [mcts(state), stationaryAgentPolicy(state)]
    sampleValidationTrajectory = SampleTrajectory(validationMaxRunningSteps, transit, isTerminal, reset, distToAction)
    validationTrajectories = [sampleValidationTrajectory(validationPolicy) for _ in range(validationSize)]
    validationData = preProcessTrajectories(validationTrajectories)

    NNModel = getModel()
    validations = []
    for iteration in range(numIterations):
        print("iteration", iteration)
        validations.append(evaluate(NNModel, validationData))
        trajectories = playToTrain(NNModel, {'iteration': iteration, 'learnRate': learningRate})
        trainData = preProcessTrajectories(trajectories)
        updatedNNModel = trainToPlay(NNModel, trainData, {'iteration': iteration, 'learnRate': learningRate})
        NNModel = updatedNNModel


    print("--------VALIDATIONS--------")
    print(validations)

if __name__ == '__main__':
    main()