import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.trajectoriesSaveLoad import loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from src.constrainedChasingEscapingEnv.policies import RandomPolicy

from src.inferChasing.continuousPolicy import ThreeAgentsPolicyForNN
from src.inferChasing.continuousTransition import TransitConstantPhysics
from src.inferChasing.inference import IsInferenceTerminal, Observe, QueryDecayedLikelihood, \
    InferOneStepLikelihood, InferContinuousChasingAndDrawDemo, softenPolicy

from visualize.initialization import initializeScreen
from visualize.inferenceVisualization import SaveImage, GetChasingRoleColor, \
    GetChasingResultColor, ColorChasingPoints, DrawContinuousInferenceResultNoPull, \
    PlotInferenceProb
from visualize.continuousVisualization import ScaleState, AdjustStateFPS,\
    DrawBackground, DrawState

import itertools as it
import pandas as pd
from pygame.color import THECOLORS
import mujoco_py as mujoco
import numpy as np

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
    wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepActionSpace = [(8, 0), (6, 6), (0, 8), (-6, 6), (-8, 0), (-6, -6), (0, -8), (6, -6)]

    numActionSpace = len(wolfActionSpace)
    numStateSpace = 12
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    # wolf NN Policy
    wolfModelPath = os.path.join(dirName, '..', 'NNModels', 'wolfNNModels',
                                 'killzoneRadius=0.5_maxRunningSteps=10_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_trainSteps=99999')
    wolfNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(wolfNNModel, wolfModelPath)
    wolfPolicy = ApproximatePolicy(wolfNNModel, wolfActionSpace)  # input state, return action distribution

    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..', 'NNModels', 'sheepNNModels',
                                  'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    sheepPolicy = ApproximatePolicy(sheepNNModel, sheepActionSpace)  # input sheepstate, return action distribution

    randomActionSpace = sheepActionSpace
    randomPolicy = RandomPolicy(randomActionSpace)

    wolfInferencePolicy = softenPolicy(wolfPolicy)
    sheepInferencePolicy = softenPolicy(sheepPolicy)

    policy = ThreeAgentsPolicyForNN(wolfInferencePolicy, sheepInferencePolicy, randomPolicy)

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsActions, physics, nextState: \
        policy(mind, state, allAgentsActions) * transition(physics, state, allAgentsActions, nextState)

    dataIndex = 1
    dataPath = os.path.join(dirName, '..', 'trainedData', 'NNleasedTrajDiffActSpace'+ str(dataIndex) + '.pickle')
    trajectory = loadFromPickle(dataPath)
    stateIndex = 0

    chasingAgents = ['sheep', 'wolf', 'random']
    chasingSpace = list(it.permutations(chasingAgents))
    pullingSpace = ['constantPhysics']
    numOfAgents = len(chasingAgents)

    actionHypoList = [list(it.permutations([x,y,z])) for x in wolfActionSpace for y in sheepActionSpace for z in randomActionSpace]
    actionHypoWithDuplicates = list(it.chain.from_iterable(actionHypoList))
    actionHypo = [tuple([tuple(action) for action in actionList]) for actionList in
                  np.unique(actionHypoWithDuplicates, axis=0).tolist()]

    iterables = [chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables, names=['mind', 'physics', 'action'])


# visualization
    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = initializeScreen(fullScreen, screenWidth, screenHeight)

    leaveEdgeSpace = 200
    lineWidth = 3
    xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
    yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]
    screenColor = THECOLORS['black']
    lineColor = THECOLORS['white']
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary,
                                    lineColor, lineWidth)
    circleSize = 10
    positionIndex = [0, 1]
    drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

    wolfColor = THECOLORS['red']
    sheepColor = THECOLORS['green']
    randomColor = THECOLORS['blue']

    wolfIndex = 'wolf'
    sheepIndex = 'sheep'
    randomIndex = 'random'

    getWolfColor = GetChasingRoleColor(wolfColor, wolfIndex)
    getSheepColor = GetChasingRoleColor(sheepColor, sheepIndex)
    getRandomColor = GetChasingRoleColor(randomColor, randomIndex)

    getRolesColor = [getWolfColor, getSheepColor, getRandomColor]
    getChasingResultColor = GetChasingResultColor(getRolesColor)
    colorChasingPoints = ColorChasingPoints(getChasingResultColor)

    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [210, 590]
    scaledYRange = [210, 590]
    scaleState = ScaleState(positionIndex, rawXRange,rawYRange, scaledXRange, scaledYRange)

    oldFPS = 5
    FPS = 60
    adjustFPS = AdjustStateFPS(oldFPS, FPS)

    imageFolderName = 'inferNNDiffActSpLeasedChasing' + str(dataIndex)
    saveImage = SaveImage(imageFolderName)
    drawInferenceResult = DrawContinuousInferenceResultNoPull(numOfAgents, inferenceIndex,
                drawState, scaleState, colorChasingPoints, adjustFPS, saveImage)

    thresholdPosterior = 1.5
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    decayParameter = 0.8
    queryLikelihood = QueryDecayedLikelihood(mindPhysicsName, decayParameter)

    inferOneStepLikelihood = InferOneStepLikelihood(inferenceIndex, getMindsPhysicsActionsJointLikelihood)

    observe = Observe(stateIndex, trajectory)
    inferContinuousChasingAndDrawDemo = InferContinuousChasingAndDrawDemo(FPS, inferenceIndex,
                                                                          isInferenceTerminal, observe, queryLikelihood,
                                                                          inferOneStepLikelihood, drawInferenceResult)

    mindsPhysicsPrior = [1 / len(inferenceIndex)] * len(inferenceIndex)
    posteriorDf = inferContinuousChasingAndDrawDemo(numOfAgents, mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotName = 'NNDiffActSpLeashedWolfInference_MemoryDecay08_SoftenPolicy'
    plotMindInferenceProb(posteriorDf, dataIndex, plotName)
    plotPhysicsInferenceProb(posteriorDf, dataIndex, plotName)


if __name__ == '__main__':
    main()