import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from exec.trajectoriesSaveLoad import loadFromPickle
from src.algorithms.mcts import Expand, ScoreChild, SelectChild, MCTS, InitializeChildren, establishPlainActionDist, \
    backup, RollOut
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunctionWithoutXPos
from src.episode import chooseGreedyAction
from src.constrainedChasingEscapingEnv.reward import HeuristicDistanceToTarget, RewardFunctionCompete
from src.constrainedChasingEscapingEnv.policies import UniformPolicy

from src.inferChasing.continuousPolicy import ThreeAgentsPolicyForMCTS
from src.inferChasing.continuousTransition import TransitConstantPhysics
from src.inferChasing.inference import IsInferenceTerminal, Observe, InferOneStep, \
    InferContinuousChasingAndDrawDemo


from visualize.initialization import initializeScreen
from visualize.inferenceVisualization import SaveImage, GetChasingRoleColor, \
    GetChasingResultColor, ColorChasingPoints, DrawContinuousInferenceResultNoPull, \
    PlotInferenceProb
from visualize.continuousVisualization import ScaleState, AdjustStateFPS,\
    DrawBackground, DrawState


import itertools as it
import pandas as pd
from pygame.color import THECOLORS
import numpy as np 
import mujoco_py as mujoco
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    # manipulated variables and other important parameters
    dirName = os.path.dirname(__file__)
    physicsDynamicsPath = os.path.join(dirName, '..', 'env', 'xmls', 'leased.xml')
    physicsModel = mujoco.load_model_from_path(physicsDynamicsPath)
    physicsSimulation = mujoco.MjSim(physicsModel)

    sheepId = 0
    wolfId = 1
    qPosIndex = [0, 1]
    getSheepQPos = GetAgentPosFromState(sheepId, qPosIndex)
    getWolfQPos = GetAgentPosFromState(wolfId, qPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepQPos, getWolfQPos)

    numSimulationFrames = 20
    transit = TransitionFunctionWithoutXPos(physicsSimulation, isTerminal, numSimulationFrames)
    transition = TransitConstantPhysics(transit)

    # policy
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    randomPolicy = UniformPolicy(actionSpace)
    transitInWolfMCTSSimulation = \
        lambda state, wolfSelfAction: transit(state, [chooseGreedyAction(randomPolicy(state)), wolfSelfAction, chooseGreedyAction(randomPolicy(state))])

    numActionSpace = len(actionSpace)
    getUniformActionPrior = lambda state: {action: 1/numActionSpace for action in actionSpace}
    initializeChildrenUniformPrior = InitializeChildren(actionSpace, transitInWolfMCTSSimulation,
                                                        getUniformActionPrior)
    expand = Expand(isTerminal, initializeChildrenUniformPrior)

    aliveBonus = -0.05
    deathPenalty = 1
    rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0.1
    maxRolloutSteps = 10
    rolloutHeuristic = HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfQPos, getSheepQPos)
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInWolfMCTSSimulation, rewardFunction, isTerminal, rolloutHeuristic)

    numSimulations = 100
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist)

    policy = ThreeAgentsPolicyForMCTS(mcts, randomPolicy)

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsActions, physics, nextState: \
        policy(mind, state, allAgentsActions) * transition(physics, state, allAgentsActions, nextState)

    dataIndex = 3
    dataPath = os.path.join(dirName, '..', 'trainedData', 'leasedMCTSTraj'+ str(dataIndex) + '.pickle')
    alltrajectory = loadFromPickle(dataPath)
    trajectory = [alltrajectory[1], alltrajectory[2]]
    # trajectory  = alltrajectory[:2]
    stateIndex = 0


    chasingAgents = ['sheep', 'wolf', 'random']
    chasingSpace = list(it.permutations(chasingAgents))
    pullingSpace = ['constantPhysics']
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

    imageFolderName = 'inferMCTSLeasedChasing' + str(dataIndex)
    saveImage = SaveImage(imageFolderName)
    drawInferenceResult = DrawContinuousInferenceResultNoPull(numOfAgents, inferenceIndex,
                drawState, scaleState, colorChasingPoints, adjustFPS, saveImage)

    thresholdPosterior = 1.5
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    mindPhysicsActionName = ['mind', 'physics', 'action']
    inferOneStep = InferOneStep(inferenceIndex, mindPhysicsActionName, getMindsPhysicsActionsJointLikelihood)

    observe = Observe(stateIndex, trajectory)
    inferContinuousChasingAndDrawDemo = InferContinuousChasingAndDrawDemo(FPS, inferenceIndex,
                                                                          isInferenceTerminal, observe, inferOneStep,
                                                                          drawInferenceResult)

    mindsPhysicsPrior = [1 / len(inferenceIndex)] * len(inferenceIndex)
    posteriorDf = inferContinuousChasingAndDrawDemo(numOfAgents, mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotMindInferenceProb(posteriorDf, dataIndex)
    plotPhysicsInferenceProb(posteriorDf, dataIndex)
#

if __name__ == '__main__':
    main()
