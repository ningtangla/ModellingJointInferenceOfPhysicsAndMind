import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.trajectoriesSaveLoad import loadFromPickle, saveToPickle
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.policies import RandomPolicy

from src.inferChasing.continuousPolicy import ThreeAgentsPolicyForNN
from src.inferChasing.continuousTransition import TransitConstantPhysics
from src.inferChasing.inference import IsInferenceTerminal, Observe, QueryDecayedLikelihoodWithParam, \
    InferOneStepLikelihoodWithParams, InferContinuousChasingAndDrawDemo, softenPolicy

import itertools as it
import pandas as pd
import mujoco_py as mujoco
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'leased.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)
    numSimulationFrames = 20

    sheepId = 0
    wolfId = 1
    qPosIndex=[0,1]
    getSheepXPos = GetAgentPosFromState(sheepId, qPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, qPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    transit = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    transition = TransitConstantPhysics(transit)

    # Neural Network
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(actionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # wolf NN Policy
    wolfModelPath = os.path.join(dirName, '..','NNModels','wolfNNModels', 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, wolfModelPath)
    wolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace) # input state, return action distribution

    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..','NNModels','sheepNNModels', 'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace) # input sheepstate, return action distribution

    randomPolicy = RandomPolicy(actionSpace)

    policy = lambda softParam: ThreeAgentsPolicyForNN(wolfPolicy, sheepPolicy, randomPolicy, softenPolicy, softParam)


    getMindsPhysicsActionsJointLikelihoodWithSoftParam = lambda softParam, mind, state, allAgentsActions, physics, nextState: \
        policy(softParam)(mind, state, allAgentsActions) * transition(physics, state, allAgentsActions, nextState)

    dataIndex = 1
    dataPath = os.path.join(dirName, '..', 'trainedData', 'NNLeasedTraj'+ str(dataIndex) + '.pickle')
    trajectory = loadFromPickle(dataPath)
    stateIndex = 0

    softPolicyParam = [0.25, 0.5, 1, 1.5, 2]
    decayMemoryParam = [0.25, 0.5, 0.75, 1]
    chasingAgents = ['sheep', 'wolf', 'random']
    chasingSpace = list(it.permutations(chasingAgents))
    pullingSpace = ['constantPhysics']
    numOfAgents = len(chasingAgents)
    actionHypo = list(it.product(actionSpace, repeat=numOfAgents))
    iterables = [softPolicyParam, decayMemoryParam, chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables,
                                                names=['softPolicy', 'decayMemory', 'mind', 'physics', 'action'])

    inferOneStepLikelihoodWithParams = InferOneStepLikelihoodWithParams(inferenceIndex,
                                                                        getMindsPhysicsActionsJointLikelihoodWithSoftParam)
    thresholdPosterior = 1.5
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, inferenceIndex)

    decayParamName = 'decayMemory'
    queryDecayedLikelihood = QueryDecayedLikelihoodWithParam(inferenceIndex, decayParamName)

    FPS = 60
    observe = Observe(stateIndex, trajectory)
    inferContinuousChasing = InferContinuousChasingAndDrawDemo(FPS, inferenceIndex, isInferenceTerminal,
                                                               observe, queryDecayedLikelihood,
                                                               inferOneStepLikelihoodWithParams)

    hypothesisSpaceSize = len(chasingSpace)* len(pullingSpace)* len(actionHypo)
    mindsPhysicsPrior = [1 / hypothesisSpaceSize] * len(inferenceIndex)
    posteriorDf = inferContinuousChasing(numOfAgents, mindsPhysicsPrior)

    saveInferenceResultPath = os.path.join(dirName, '..', 'trainedData', 'posteriorResult20Params' + str(dataIndex) + '.pickle')
    saveToPickle(posteriorDf, saveInferenceResultPath)

#visualization
    # posteriorDf = loadFromPickle(saveInferenceResultPath)

    fig = plt.figure(dpi=1000)
    nCols = len(decayMemoryParam)
    nRows = len(softPolicyParam)
    subplotIndex = 1
    axs = []
    fig.set_size_inches(7.5, 6)
    fig.suptitle('evaluateInferenceParameters: SoftParam: ' + str(softPolicyParam)+ ', DecayParam: '+ str(decayMemoryParam), fontsize = 8)

    for softParam, softGrp in posteriorDf.groupby('softPolicy'):
        softGrp.index = softGrp.index.droplevel('softPolicy')

        for decayParam, decayGroup in softGrp.groupby('decayMemory'):
            decayGroup.index = decayGroup.index.droplevel('decayMemory')

            ax = fig.add_subplot(nRows, nCols, subplotIndex)  # = subplot
            axs.append(ax)
            subplotIndex += 1
            resultDf = decayGroup.groupby('mind').sum()
            resultDf.T.plot(ax=ax)
            plt.ylim([0, 1])

            ax.set_title('soft = '+ str(softParam) + ' decay = '+ str(decayParam), fontsize = 5)
            ax.set_ylabel('posterior', fontsize = 5)
            ax.set_xlabel('timeStep', fontsize = 5)
            ax.tick_params(axis='both', which='major', labelsize=5)
            ax.tick_params(axis='both', which='minor', labelsize=5)

            plt.legend(loc='best', prop={'size': 2})
            # print('softParam = '+ str(softParam) + ' decayParam = '+ str(decayParam))
            # print(resultDf)

    plotPath = os.path.join(dirName, '..', 'demo')
    plt.savefig(os.path.join(plotPath, 'inferenceParameterEvaluation20Params'+ str(dataIndex)))


if __name__ == '__main__':
    main()