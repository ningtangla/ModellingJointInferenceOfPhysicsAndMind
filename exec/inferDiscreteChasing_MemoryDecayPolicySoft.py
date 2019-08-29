import sys
import os
sys.path.append(os.path.join('..'))

import itertools
import pandas as pd
import numpy as np
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.constrainedChasingEscapingEnv.policies import RandomPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

from src.inferChasing.discreteGridPolicy import ActHeatSeeking, \
    HeatSeekingPolicy, WolfPolicy, SheepPolicy, MasterPolicy
from src.inferChasing.discreteGridTransition import StayWithinBoundary, PulledForceLikelihood, \
    PulledTransition, NoPullTransition
from src.inferChasing.inference import IsInferenceTerminal, ObserveStateOnly, InferOneStepLikelihood, \
    InferDiscreteChasingWithMemoryDecayAndDrawDemo, QueryDecayedLikelihood, softenPolicy

from visualize.inferenceVisualization import SaveImage, GetChasingRoleColor, \
    GetChasingResultColor, ColorChasingPoints, AdjustPullingLineWidth, \
    DrawInferenceResultWithPull, PlotInferenceProb
from visualize.initialization import initializeScreen
from visualize.discreteGridVisualization import checkDuplicates, ModifyOverlappingPoints, DrawGrid,\
    DrawCirclesAndLines


def main():
    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2
    actHeatSeeking = ActHeatSeeking(actionSpace, lowerBoundAngle, upperBoundAngle,
                                    computeAngleBetweenVectors)

    rationalityParam = 0.9
    heatSeekingPolicy = HeatSeekingPolicy(rationalityParam, actHeatSeeking)
    softParameter = 1

    softenedHeatSeeking = softenPolicy(heatSeekingPolicy, softParameter)

    positionIndex = [0, 1]
    getAgentPosition = lambda agentID, state: GetAgentPosFromState(agentID, positionIndex)(state)

    wolfPolicy = WolfPolicy(getAgentPosition, softenedHeatSeeking)
    sheepPolicy = SheepPolicy(getAgentPosition, softenedHeatSeeking)

    uniformPolicy = RandomPolicy(actionSpace)
    masterPolicy = MasterPolicy(uniformPolicy)

    policyList = [wolfPolicy, sheepPolicy, masterPolicy]
    policy = lambda mind, state, allAgentsAction: np.product(
        [agentPolicy(mind, state, allAgentsAction) for agentPolicy in policyList])


    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    getPulledAgentForceLikelihood = PulledForceLikelihood(forceSpace, lowerBoundAngle,
                                                          upperBoundAngle, computeAngleBetweenVectors)
    gridSize = (10, 10)
    lowerBoundary = 1
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)

    pulledTransition = PulledTransition(getAgentPosition, getPulledAgentForceLikelihood,
                                        stayWithinBoundary)
    noPullTransition = NoPullTransition(getAgentPosition, stayWithinBoundary)
    transitionList = [pulledTransition, noPullTransition]
    transition = lambda physics, state, allAgentsAction, nextState: \
        np.product([agentTransition(physics, state, allAgentsAction, nextState)
                    for agentTransition in transitionList])

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsAction, physics, nextState: \
        policy(mind, state, allAgentsAction) * transition(physics, state, allAgentsAction, nextState)


# visualization
    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = initializeScreen(fullScreen, screenWidth, screenHeight)

    gridNumberX, gridNumberY = gridSize
    gridPixelSize = min(screenHeight// gridNumberX, screenWidth// gridNumberY)
    modificationRatio = 3
    modifyOverlappingPoints = ModifyOverlappingPoints(gridPixelSize, checkDuplicates, modificationRatio)
    drawCirclesAndLines = DrawCirclesAndLines(modifyOverlappingPoints)
    drawGrid = DrawGrid(screen, gridSize, gridPixelSize)

    wolfColor = THECOLORS['red']
    sheepColor = THECOLORS['green']
    masterColor = THECOLORS['blue']

    wolfIndex = 'wolf'
    sheepIndex = 'sheep'
    masterIndex = 'master'

    getWolfColor = GetChasingRoleColor(wolfColor, wolfIndex)
    getSheepColor = GetChasingRoleColor(sheepColor, sheepIndex)
    getMasterColor = GetChasingRoleColor(masterColor, masterIndex)
    getRolesColor = [getWolfColor, getSheepColor, getMasterColor]
    getChasingResultColor = GetChasingResultColor(getRolesColor)

    colorChasingPoints = ColorChasingPoints(getChasingResultColor)
    adjustPullingLineWidth = AdjustPullingLineWidth()

    chasingAgents = ['wolf', 'sheep', 'master']
    chasingSpace = list(itertools.permutations(chasingAgents))
    pullingAgents = ['pulled', 'noPull', 'pulled']
    pullingSpaceArray = np.unique(list(itertools.permutations(pullingAgents)), axis=0)
    pullingSpace = [tuple(pullingPair) for pullingPair in pullingSpaceArray.tolist()]
    actionHypo = list(itertools.product(actionSpace, actionSpace, actionSpace))
    iterables = [chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables, names=['mind', 'physics', 'action'])

    drawInferenceResult = DrawInferenceResultWithPull(inferenceIndex, gridPixelSize,
                                                      drawGrid, drawCirclesAndLines,
                                                      colorChasingPoints, adjustPullingLineWidth)

    thresholdPosterior = 1.5
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    inferOneStepLikelihood = InferOneStepLikelihood(inferenceIndex, getMindsPhysicsActionsJointLikelihood)

    trajectory = [[(6, 2), (9, 2), (5, 4)], [(7, 3), (10, 2), (6, 3)], [(6, 2), (10, 2), (7, 2)], [(8, 2), (10, 2), (6, 1)], [(8, 2), (10, 2), (7, 1)], [(8, 2), (10, 3), (7, 1)], [(9, 1), (10, 3), (8, 2)], [(8, 2), (10, 3), (8, 2)], [(8, 3), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (8, 1)], [(10, 1), (10, 4), (7, 2)], [(9, 2), (10, 5), (9, 2)], [(9, 3), (9, 5), (9, 3)], [(9, 4), (9, 6), (10, 3)], [(10, 5), (9, 7), (10, 3)], [(9, 4), (10, 7), (10, 4)], [(10, 5), (10, 7), (10, 4)], [(10, 5), (10, 8), (9, 5)], [(9, 6), (10, 9), (10, 4)], [(10, 7), (10, 9), (8, 4)], [(9, 8), (10, 10), (9, 3)], [(10, 7), (10, 10), (10, 4)], [(10, 7), (10, 10), (9, 5)], [(10, 7), (10, 10), (8, 6)], [(9, 8), (10, 10), (10, 6)], [(9, 8), (10, 10), (10, 6)], [(10, 9), (10, 9), (9, 7)]]
    observe = ObserveStateOnly(trajectory)

    dataIndex = 0
    imageFolderName = 'demo' + str(dataIndex)
    saveImage = SaveImage(imageFolderName)
    FPS = 60

    decayParameter = 0.8
    queryLikelihood = QueryDecayedLikelihood(mindPhysicsName, decayParameter)

    inferDiscreteChasingAndDrawDemo = InferDiscreteChasingWithMemoryDecayAndDrawDemo(FPS,
                 inferenceIndex, isInferenceTerminal, observe, inferOneStepLikelihood, queryLikelihood,
                 drawInferenceResult, saveImage)

    mindsPhysicsPrior = [1/ len(inferenceIndex)] * len(inferenceIndex)

    posteriorDf = inferDiscreteChasingAndDrawDemo(mindsPhysicsPrior)

    plotMindInferenceProb = PlotInferenceProb('timeStep', 'mindPosterior', 'mind')
    plotPhysicsInferenceProb = PlotInferenceProb('timeStep', 'physicsPosterior', 'physics')

    plotName = 'Discrete3AgentsPullInference_MemoryDecay08_SoftenPolicy'
    plotMindInferenceProb(posteriorDf, dataIndex, plotName)
    plotPhysicsInferenceProb(posteriorDf, dataIndex, plotName)



if __name__ == '__main__':
    main()
