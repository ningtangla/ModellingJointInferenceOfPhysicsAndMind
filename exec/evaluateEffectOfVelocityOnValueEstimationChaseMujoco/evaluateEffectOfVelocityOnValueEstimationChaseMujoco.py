import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
import pandas as pd
import math
import random
from collections import OrderedDict
from matplotlib import pyplot as plt

from src.neuralNetwork.policyValueNet import GenerateModel, ApproximateValueFunction, restoreVariables
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle, loadFromPickle
from src.constrainedChasingEscapingEnv.envMujoco import WithinBounds
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeVectorNorm
from exec.evaluationFunctions import conditionDfFromParametersDict


class GenerateInitQPosTwoAgentsGivenDistance:
    def __init__(self, qMin, qMax, withinBounds):
        self.qMin = qMin
        self.qMax = qMax
        self.withinBounds = withinBounds

    def __call__(self, initDistance):
        while(True):
            agent0QPos = np.random.uniform(self.qMin, self.qMax, 2)
            angle = 2 * math.pi * random.random()
            agent1QPos = (agent0QPos[0] + math.cos(angle)*initDistance, agent0QPos[1] + math.sin(angle)*initDistance)
            initQPos = np.concatenate((agent0QPos, agent1QPos))
            if self.withinBounds(initQPos):
                break

        return initQPos


class GetInitStateFromWolfVelocityDirection:
    def __init__(self, sheepQPosIndex, wolfQPosIndex, computeVectorNorm, wolfVelocityMagnitude):
        self.sheepQPosIndex = sheepQPosIndex
        self.wolfQPosIndex = wolfQPosIndex
        self.computeVectorNorm = computeVectorNorm
        self.wolfVelocityMagnitude = wolfVelocityMagnitude

    def __call__(self, initQPos, initVelocityDirection):
        initQPos = np.asarray(initQPos)
        sheepQPos = initQPos[self.sheepQPosIndex]
        wolfQPos = initQPos[self.wolfQPosIndex]
        wolfVelocityRaw = sheepQPos-wolfQPos
        velocityNorm = self.computeVectorNorm(wolfVelocityRaw)
        if velocityNorm != 0:
            wolfVelocity = wolfVelocityRaw * self.wolfVelocityMagnitude / velocityNorm
        else:
            wolfVelocity = wolfVelocityRaw
        sheepVelocity = (0, 0)

        if initVelocityDirection == 'away':
            wolfVelocity *= -1
        elif initVelocityDirection == 'stationary':
            wolfVelocity *= 0

        sheepState = np.concatenate((sheepQPos, sheepQPos, sheepVelocity))
        wolfState = np.concatenate((wolfQPos, wolfQPos, wolfVelocity))
        initState = np.asarray([sheepState, wolfState])

        return initState


class ComputeMeanValue:
    def __init__(self, numSamples, approximateValue, getQPosSavePath, generateInitQPos, saveInitQPos, loadInitQPos,
                 getInitState):
        self.numSamples = numSamples
        self.approximateValue = approximateValue
        self.getQPosSavePath = getQPosSavePath
        self.generateInitQPos = generateInitQPos
        self.saveInitQPos = saveInitQPos
        self.loadInitQPos = loadInitQPos
        self.getInitState = getInitState

    def __call__(self, oneConditionDf):
        initDistance = oneConditionDf.index.get_level_values('initDistance')[0]
        QPosSavePath = self.getQPosSavePath(initDistance)
        if not os.path.isfile(QPosSavePath):
            allInitQPos = [self.generateInitQPos(initDistance) for _ in range(self.numSamples)]
            self.saveInitQPos(allInitQPos, QPosSavePath)
        else:
            allInitQPos = self.loadInitQPos(QPosSavePath)
        initVelocityDirection = oneConditionDf.index.get_level_values('initVelocityDirection')[0]
        allInitState = [self.getInitState(initQPos, initVelocityDirection) for initQPos in allInitQPos]
        allValues = [self.approximateValue(state) for state in allInitState]
        meanValue = np.mean(allValues)

        return pd.Series({'value': meanValue})

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['initDistance'] = [4, 8]#[2, 4, 8, 16]
    manipulatedVariables['initVelocityDirection'] = ['towards', 'away', 'stationary']
    numSamples = 10#200

    toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)
    levelNames = list(manipulatedVariables.keys())

    # Neural Network
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)
    NNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)

    # restore model
    NNModelPath = NNModelSavePath = '/Users/nishadsinghi/ModellingJointInferenceOfPhysicsAndMind/data/evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco/trainedNNModels/killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999'
    restoreVariables(NNModel, NNModelPath)
    approximateValue = ApproximateValueFunction(NNModel)

    # path for saving init q positions
    qPosInitSaveDirectory = os.path.join('..', '..', 'data', 'evaluateEffectOfVelocityOnValueEstimationChaseMujoco',
                                         'initQPos')
    if not os.path.exists(qPosInitSaveDirectory):
        os.makedirs(qPosInitSaveDirectory)
    qPosInitSaveParameters = {'numSamples': numSamples}
    qPosInitSaveExtension = '.pickle'
    getQPosInitSavePath = GetSavePath(qPosInitSaveDirectory, qPosInitSaveExtension, qPosInitSaveParameters)

    # generate init q positions
    withinBounds = WithinBounds((-9.7, -9.7, -9.7, -9.7), (9.7, 9.7, 9.7, 9.7))
    generateInitQPos = GenerateInitQPosTwoAgentsGivenDistance(-9.7, 9.7, withinBounds)

    # get init state from q position and velocity direction
    getInitState = GetInitStateFromWolfVelocityDirection([0, 1], [2, 3], computeVectorNorm, 4)

    # do the evaluation
    computeMeanValue = ComputeMeanValue(numSamples, approximateValue, getQPosInitSavePath, generateInitQPos,
                                        saveToPickle, loadFromPickle, getInitState)
    valueDf = toSplitFrame.groupby(levelNames).apply(computeMeanValue)

    # plot
    fig = plt.figure()
    axForDraw = fig.add_subplot(1, 1, 1)

    for direction, grp in valueDf.groupby('initVelocityDirection'):
        grp.index = grp.index.droplevel('initVelocityDirection')
        grp.plot(marker='o', ax=axForDraw, label=direction)

    plt.ylabel('estimated value')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()