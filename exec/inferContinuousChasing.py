import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.inferChasing.continuousPolicy import Policy, RandomPolicy
from src.inferChasing.continuousTransition import Transition
from src.inferChasing.inference import IsInferenceTerminal, Observe, InferOneStep, \
    InferContinuousChasingAndDrawDemo

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy
from exec.SheepAndWolfNNOnlyPolicies.policy import AgentPolicy
from exec.trajectoriesSaveLoad import loadFromPickle

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


def main():
    # transition function
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'twoAgents.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    numSimulationFrames = 20
    transitAgents = TransitionFunction(physicsSimulation, isTerminal, numSimulationFrames)
    transition = Transition(transitAgents)

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
    approximateWolfPolicy = ApproximatePolicy(wolfNNModel, actionSpace)
    wolfPolicy = AgentPolicy(approximateWolfPolicy)


    # sheep NN Policy
    sheepModelPath = os.path.join(dirName, '..','NNModels','sheepNNModels', 'killzoneRadius=2_maxRunningSteps=25_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=8_rolloutHeuristicWeight=0.1_trainSteps=99999')
    sheepNNModel = generateModel(sharedWidths, actionLayerWidths, valueLayerWidths)
    restoreVariables(sheepNNModel, sheepModelPath)
    approximateSheepPolicy = ApproximatePolicy(sheepNNModel, actionSpace)
    sheepPolicy = AgentPolicy(approximateSheepPolicy)

    # random Policy
    randomPolicy = RandomPolicy(actionSpace)
    policy = Policy(wolfPolicy, sheepPolicy, randomPolicy)

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsActions, physics, nextState: \
        policy(mind, state, allAgentsActions) * transition(physics, state, allAgentsActions, nextState)

    dataIndex = 5
    dataPath = os.path.join(dirName, '..', 'trainedData', 'trajectory'+ str(dataIndex) + '.pickle')
    trajectory = loadFromPickle(dataPath)
    stateIndex = 0
    observe = Observe(stateIndex, trajectory)

    chasingAgents = ['wolf', 'sheep']
    chasingSpace = list(it.permutations(chasingAgents))
    chasingSpace.append(('random', 'random'))
    pullingSpace = ['noPull']
    numOfAgents = len(chasingAgents)
    actionHypo = list(it.product(actionSpace, repeat=numOfAgents))
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
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    circleSize = 10
    positionIndex = [0, 1]
    drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

    wolfColor = THECOLORS['red']
    sheepColor = THECOLORS['green']
    wolfIndex = 'wolf'
    sheepIndex = 'sheep'
    getWolfColor = GetChasingRoleColor(wolfColor, wolfIndex)
    getSheepColor = GetChasingRoleColor(sheepColor, sheepIndex)
    getRolesColor = [getWolfColor, getSheepColor]
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

    imageFolderName = 'continuousDemo' + str(dataIndex)
    saveImage = SaveImage(imageFolderName)
    drawInferenceResult = DrawContinuousInferenceResultNoPull(inferenceIndex, drawState, scaleState, colorChasingPoints, adjustFPS, saveImage)

    thresholdPosterior = 1.5
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    mindPhysicsActionName = ['mind', 'physics', 'action']
    inferOneStep = InferOneStep(inferenceIndex, mindPhysicsActionName, getMindsPhysicsActionsJointLikelihood)

    inferContinuousChasingAndDrawDemo = InferContinuousChasingAndDrawDemo(FPS, inferenceIndex,
                 isInferenceTerminal, observe, inferOneStep, drawInferenceResult)
    mindsPhysicsPrior = [1/ len(inferenceIndex)] * len(inferenceIndex)
    posteriorDf = inferContinuousChasingAndDrawDemo(mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotMindInferenceProb(posteriorDf, dataIndex)
    plotPhysicsInferenceProb(posteriorDf, dataIndex)

    # posteriorDf.to_csv("posterior.csv")

if __name__ == '__main__':
    main()

