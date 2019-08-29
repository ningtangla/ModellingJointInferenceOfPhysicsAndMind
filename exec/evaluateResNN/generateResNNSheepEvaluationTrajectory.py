import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import json
from collections import OrderedDict
import pickle
import pandas as pd
import time
import mujoco_py as mujoco
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from src.constrainedChasingEscapingEnv.envMujoco import ResetUniform, IsTerminal, TransitionFunction
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
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, HeuristicDistanceToTarget
from exec.preProcessing import AccumulateRewards

class GenerateModelWolf:
    def __init__(self, numStateSpace, numActionSpace, regularizationFactor=0, valueRelativeErrBound=0.01, seed=128):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.regularizationFactor = regularizationFactor
        self.valueRelativeErrBound = valueRelativeErrBound
        self.seed = seed

    def __call__(self, sharedWidths, actionLayerWidths, valueLayerWidths, summaryPath="./tbdata"):
        print("Generating NN with shared layers: {}, action layers: {}, value layers: {}"
              .format(sharedWidths, actionLayerWidths, valueLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                tf.add_to_collection("inputs", states_)

            with tf.name_scope("groundTruths"):
                groundTruthAction_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="action")
                groundTruthValue_ = tf.placeholder(tf.float32, [None, 1], name="value")
                tf.add_to_collection("groundTruths", groundTruthAction_)
                tf.add_to_collection("groundTruths", groundTruthValue_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                actionLossCoef_ = tf.constant(1, dtype=tf.float32)
                valueLossCoef_ = tf.constant(1, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("lossCoefs", actionLossCoef_)
                tf.add_to_collection("lossCoefs", valueLossCoef_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)

            with tf.variable_scope("shared"):
                activation_ = states_
                for i in range(len(sharedWidths)):
                    fcLayer = tf.layers.Dense(units=sharedWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                sharedOutput_ = tf.identity(activation_, name="output")

            with tf.variable_scope("action"):
                activation_ = sharedOutput_
                for i in range(len(actionLayerWidths)):
                    fcLayer = tf.layers.Dense(units=actionLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                outputFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(actionLayerWidths) + 1))
                actionOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{actionOutputLayerActivation_.name}"],
                                      actionOutputLayerActivation_)

            with tf.name_scope("actionOutputs"):
                actionDistributions_ = tf.nn.softmax(actionOutputLayerActivation_, name="distributions")
                actionIndices_ = tf.argmax(actionDistributions_, axis=1, name="indices")
                tf.add_to_collection("actionDistributions", actionDistributions_)
                tf.add_to_collection("actionIndices", actionIndices_)

            with tf.variable_scope("value"):
                activation_ = sharedOutput_
                for i in range(len(valueLayerWidths)):
                    fcLayer = tf.layers.Dense(units=valueLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)

                outputFCLayer = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(valueLayerWidths) + 1))
                valueOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{valueOutputLayerActivation_.name}"],
                                      valueOutputLayerActivation_)

            with tf.name_scope("valueOutputs"):
                values_ = tf.identity(valueOutputLayerActivation_, name="values")
                tf.add_to_collection("values", values_)

            with tf.name_scope("evaluate"):
                with tf.name_scope("action"):
                    crossEntropy_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionOutputLayerActivation_,
                                                                               labels=groundTruthAction_)
                    actionLoss_ = tf.reduce_mean(crossEntropy_, name='loss')
                    tf.add_to_collection("actionLoss", actionLoss_)
                    actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

                    groundTruthActionIndices_ = tf.argmax(groundTruthAction_, axis=1)
                    actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, groundTruthActionIndices_), tf.float32), name="accuracy")
                    tf.add_to_collection("actionAccuracy", actionAccuracy_)
                    actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

                with tf.name_scope("value"):
                    valueLoss_ = tf.losses.mean_squared_error(groundTruthValue_, values_)
                    tf.add_to_collection("valueLoss", valueLoss_)
                    valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

                    relativeErrorBound_ = tf.constant(self.valueRelativeErrBound)
                    relativeValueError_ = tf.cast((tf.abs((values_ - groundTruthValue_) / groundTruthValue_)), relativeErrorBound_.dtype)
                    valueAccuracy_ = tf.reduce_mean(tf.cast(tf.less(relativeValueError_, relativeErrorBound_), tf.float64), name="accuracy")
                    tf.add_to_collection("valueAccuracy", valueAccuracy_)
                    valueAccuracySummary = tf.summary.scalar("valueAccuracy", valueAccuracy_)

                with tf.name_scope("regularization"):
                    l2RegularizationLoss_ = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]),
                                                        self.regularizationFactor, name="l2RegLoss")
                    tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)

                loss_ = tf.add_n([actionLossCoef_*actionLoss_, valueLossCoef_*valueLoss_, l2RegularizationLoss_], name="loss")
                tf.add_to_collection("loss", loss_)
                lossSummary = tf.summary.scalar("loss", loss_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                gradVarPairs_ = optimizer.compute_gradients(loss_)
                trainOp = optimizer.apply_gradients(gradVarPairs_)
                tf.add_to_collection("trainOp", trainOp)

                with tf.name_scope("inspectGrad"):
                    for grad_, var_ in gradVarPairs_:
                        keyPrefix = "weightGradient" if "kernel" in var_.name else "biasGradient"
                        tf.add_to_collection(f"{keyPrefix}/{var_.name}", grad_)
                    gradients_ = [tf.reshape(grad_, [1, -1]) for (grad_, _) in gradVarPairs_]
                    allGradTensor_ = tf.concat(gradients_, 1)
                    allGradNorm_ = tf.norm(allGradTensor_)
                    tf.add_to_collection("allGradNorm", allGradNorm_)
                    tf.summary.histogram("allGradients", allGradTensor_)
                    tf.summary.scalar("allGradNorm", allGradNorm_)
            fullSummary = tf.summary.merge_all()
            evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary,
                                            actionAccuracySummary, valueAccuracySummary])
            tf.add_to_collection("summaryOps", fullSummary)
            tf.add_to_collection("summaryOps", evalSummary)

            # if summaryPath is not None:
            #     # trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
            #     # testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
            #     tf.add_to_collection("writers", trainWriter)
            #     tf.add_to_collection("writers", testWriter)

            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model

def main():
    dirName = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'evaluateResNN',
                                       'evaluateTrajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    trainMaxRunningSteps = 25
    trainNumSimulations = 100
    killzoneRadius = 2
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

        isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

        numSimulationFrames = 20
        transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)

        alivePenalty = 0.05
        deathBonus = -1
        rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

        # neural network init and save path
        numStateSpace = 12
        numActionSpace = 8
        regularizationFactor = 1e-2
        resBlockSize = 2
        initialization = 'uniform'
        nnStructure = ((256,) * 17, (256,), (256,))
        sharedWidths, actionLayerWidths, valueLayerWidths = nnStructure
        generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

        NNFixedParameters = {'agentId': sheepId, 'maxRunningSteps': trainMaxRunningSteps, 'numSimulations': trainNumSimulations}
        dirName = os.path.dirname(__file__)
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', 'data', 'evaluateResNN',
                                            'trainedResSheepModels','res2depth17')
        NNModelSaveExtension = ''
        getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNFixedParameters)

        dropOutRate = float(parametersForTrajectoryPath['dropOutRate'])

        sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths, resBlockSize=resBlockSize,
                          initialization=initialization, dropoutRate=dropOutRate)


        # generate a set of starting conditions to maintain consistency across all the conditions
        evalQPosInitNoise = 0
        evalQVelInitNoise = 0
        qVelInit = [0, 0, 0, 0]

        getResetFromQPosInitDummy = lambda qPosInit: ResetUniform(physicsSimulation, qPosInit, qVelInit, numAgents, evalQPosInitNoise, evalQVelInitNoise)

        evalNumTrials = 300
        generateInitQPos = GenerateInitQPosUniform(-9.7, 9.7, isTerminal, getResetFromQPosInitDummy)
        evalAllQPosInit = [generateInitQPos() for _ in range(evalNumTrials)]
        evalAllQVelInit = np.random.uniform(-8, 8, (evalNumTrials, 4))
        getResetFromTrial = lambda trial: ResetUniform(physicsSimulation, evalAllQPosInit[trial], evalAllQVelInit[trial],
                                                       numAgents, evalQPosInitNoise, evalQVelInitNoise)
        evalMaxRunningSteps = 25
        getSampleTrajectory = lambda trial: SampleTrajectory(evalMaxRunningSteps, transit, isTerminal,
                                                             getResetFromTrial(trial), chooseGreedyAction)
        allSampleTrajectories = [getSampleTrajectory(trial) for trial in range(evalNumTrials)]

        # save evaluation trajectories
        manipulatedVariables = json.loads(sys.argv[1])
        generateModelWolf = GenerateModelWolf(numStateSpace, numActionSpace, regularizationFactor)
        modelPath = getNNModelSavePath(manipulatedVariables)
        restoredSheepModel = restoreVariables(sheepNNModel, modelPath)
        sheepPolicy = ApproximatePolicy(restoredSheepModel, actionSpace)
        resBlockWolf = 0
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        initWolfNNModel = generateModelWolf(sharedWidths, actionLayerWidths, valueLayerWidths)
        wolfNNModelPath= os.path.join(dirName, '..', '..', 'data', 'evaluateSupervisedLearning',
                                            'wolfModel','killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
        restoredWolfModel = restoreVariables(initWolfNNModel, wolfNNModelPath)

        wolfPolicy = ApproximatePolicy(restoredWolfModel, actionSpace)
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

        beginTime = time.time()
        trajectories = [sampleTrajectory(policy) for sampleTrajectory in allSampleTrajectories[startSampleIndex:endSampleIndex]]
        processTime = time.time() - beginTime
        saveToPickle(trajectories, trajectorySavePath)
        restoredSheepModel.close()
        restoredWolfModel.close()


if __name__ == '__main__':
    main()
