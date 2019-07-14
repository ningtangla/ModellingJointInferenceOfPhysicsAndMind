import sys
import os
import itertools as it
import pandas as pd
from pygame.color import THECOLORS

sys.path.append(os.path.join('..', 'src', 'inferContinuousChasing'))
sys.path.append(os.path.join('..', 'visualize'))

from continuousPolicy import Policy
from continuousTransition import Transition
from inference import IsInferenceTerminal, Observe, InferOneStep, \
    InferDiscreteChasingAndDrawDemo

from initialization import initializeScreen
from inferenceVisualization import SaveImage, GetChasingRoleColor, \
    GetChasingResultColor, ColorChasingPoints, DrawContinuousInferenceResultNoPull, \
    PlotInferenceProb
from continuousVisualization import DrawBackground, DrawState


def main():
    wolfPolicy = ___
    sheepPolicy = __
    randomPolicy = ____
    policy = Policy(wolfPolicy, sheepPolicy, randomPolicy)

    transitDeterministically = ___
    transition = Transition(transitDeterministically)

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsAction, physics, nextState: \
        policy(mind, state, allAgentsAction) * transition(physics, state, allAgentsAction, nextState)

    positionIndex = [0, 1]
    trajectory = ___
    observe = Observe(positionIndex, trajectory)

    chasingAgents = ['wolf', 'sheep']
    chasingSpace = list(it.permutations(chasingAgents))
    chasingSpace.append(('random', 'random'))

    pullingSpace = ['noPull']

    actionSpace = ___
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

    drawInferenceResult = DrawContinuousInferenceResultNoPull(inferenceIndex, drawState, colorChasingPoints)

    thresholdPosterior = 1
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    mindPhysicsActionName = ['mind', 'physics', 'action']
    inferOneStep = InferOneStep(inferenceIndex, mindPhysicsActionName, getMindsPhysicsActionsJointLikelihood)

    trajectory = [[(6, 2), (9, 2), (5, 4)], [(7, 3), (10, 2), (6, 3)], [(6, 2), (10, 2), (7, 2)], [(8, 2), (10, 2), (6, 1)], [(8, 2), (10, 2), (7, 1)], [(8, 2), (10, 3), (7, 1)], [(9, 1), (10, 3), (8, 2)], [(8, 2), (10, 3), (8, 2)], [(8, 3), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (8, 1)], [(10, 1), (10, 4), (7, 2)], [(9, 2), (10, 5), (9, 2)], [(9, 3), (9, 5), (9, 3)], [(9, 4), (9, 6), (10, 3)], [(10, 5), (9, 7), (10, 3)], [(9, 4), (10, 7), (10, 4)], [(10, 5), (10, 7), (10, 4)], [(10, 5), (10, 8), (9, 5)], [(9, 6), (10, 9), (10, 4)], [(10, 7), (10, 9), (8, 4)], [(9, 8), (10, 10), (9, 3)], [(10, 7), (10, 10), (10, 4)], [(10, 7), (10, 10), (9, 5)], [(10, 7), (10, 10), (8, 6)], [(9, 8), (10, 10), (10, 6)], [(9, 8), (10, 10), (10, 6)], [(10, 9), (10, 9), (9, 7)]]
    observe = Observe(positionIndex, trajectory)

    imageFolderName = 'continuousDemo'
    saveImage = SaveImage(imageFolderName)

    FPS = 60
    inferDiscreteChasingAndDrawDemo = InferDiscreteChasingAndDrawDemo(FPS, inferenceIndex,
                 isInferenceTerminal, observe, inferOneStep, drawInferenceResult, saveImage)

    mindsPhysicsPrior = [1/ len(inferenceIndex)] * len(inferenceIndex)
    posteriorDf = inferDiscreteChasingAndDrawDemo(mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotMindInferenceProb(posteriorDf)
    plotPhysicsInferenceProb(posteriorDf)

