import sys
import os
sys.path.append(os.path.join('..', 'src', 'inferDiscreteGridChasing'))
sys.path.append(os.path.join('..', 'src', 'constrainedChasingEscapingEnv'))
sys.path.append(os.path.join('..', 'visualize'))

import itertools
import pandas as pd
import numpy as np

from analyticGeometryFunctions import computeAngleBetweenVectors
from policyLikelihood import UniformPolicy, ActHeatSeeking, \
    HeatSeekingPolicy, WolfPolicy, SheepPolicy, MasterPolicy
from transitionLikelihood import StayWithinBoundary, PulledForceLikelihood, \
    PulledTransition, NoPullTransition
from inference import IsInferenceTerminal, saveImage, Observe, InferOneStep, InferDiscreteChasingAndDrawDemo
from state import GetAgentPosFromState
from discreteGridInferenceVisualization import checkDuplicates, ModifyOverlappingPoints, DrawCirclesAndLines, \
    InitializeGame, DrawGrid, GetChasingRoleColor, GetChasingResultColor, ColorChasingPoints, \
    AdjustPullingLineWidth, DrawInferenceResult


def main():
    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2
    actHeatSeeking = ActHeatSeeking(actionSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors)

    rationalityParam = 0.9
    heatSeekingPolicy = HeatSeekingPolicy(rationalityParam, actHeatSeeking)

    positionIndex = [0, 1]
    getAgentPosition = lambda agentID, state: GetAgentPosFromState(agentID, positionIndex)(state)

    wolfPolicy = WolfPolicy(getAgentPosition, heatSeekingPolicy)
    sheepPolicy = SheepPolicy(getAgentPosition, heatSeekingPolicy)

    uniformPolicy = UniformPolicy(actionSpace)
    masterPolicy = MasterPolicy(uniformPolicy)
    
    policyList = [wolfPolicy, sheepPolicy, masterPolicy]
    policy = lambda mind, state, allAgentsAction: np.product(
        [agentPolicy(mind, state, allAgentsAction) for agentPolicy in policyList])


    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    getPulledAgentForceLikelihood = PulledForceLikelihood(forceSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors)
    gridSize = (10, 10)
    lowerBoundary = 1
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)

    pulledTransition = PulledTransition(getAgentPosition, getPulledAgentForceLikelihood, stayWithinBoundary)
    noPullTransition = NoPullTransition(getAgentPosition, stayWithinBoundary)
    transitionList = [pulledTransition, noPullTransition]
    transition = lambda physics, state, allAgentsAction, nextState: \
        np.product([agentTransition(physics, state, allAgentsAction, nextState) for agentTransition in transitionList])

    getMindsPhysicsActionsJointLikelihood = lambda mind, state, allAgentsAction, physics, nextState: \
        policy(mind, state, allAgentsAction) * transition(physics, state, allAgentsAction, nextState)


# visualization
    screenWidth = 800
    screenHeight = 800
    caption= "Game"
    initializeGame = InitializeGame(screenWidth, screenHeight, caption)

    modificationRatio = 3
    gridNumberX, gridNumberY = gridSize
    gridPixelSize = min(screenHeight// gridNumberX, screenWidth// gridNumberY)
    modifyOverlappingPoints = ModifyOverlappingPoints(gridPixelSize, modificationRatio, checkDuplicates)

    pointExtendTime = 100
    FPS = 60
    pointWidth = 10
    BLACK = (  0,   0,   0)
    lineColor = BLACK
    drawCirclesAndLines = DrawCirclesAndLines(pointExtendTime, FPS, pointWidth, lineColor, modifyOverlappingPoints)

    WHITE = (255, 255, 255)
    backgroundColor= WHITE
    gridColor = BLACK
    gridLineWidth = 3
    drawGrid = DrawGrid(gridSize, gridPixelSize, backgroundColor, gridColor, gridLineWidth)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    wolfColor = RED
    sheepColor = GREEN
    masterColor = BLUE

    wolfIndex = 'wolf'
    sheepIndex = 'sheep'
    masterIndex = 'master'

    getWolfColor = GetChasingRoleColor(wolfColor, wolfIndex)
    getSheepColor = GetChasingRoleColor(sheepColor, sheepIndex)
    getMasterColor = GetChasingRoleColor(masterColor, masterIndex)
    getChasingResultColor = GetChasingResultColor(getWolfColor, getSheepColor, getMasterColor)

    colorChasingPoints = ColorChasingPoints(getChasingResultColor)

    minWidth = 1
    maxWidth = 5
    adjustPullingLineWidth = AdjustPullingLineWidth(minWidth, maxWidth)

    chasingAgents = ['wolf', 'sheep', 'master']
    chasingSpace = list(itertools.permutations(chasingAgents))
    pullingAgents = ['pulled', 'noPull', 'pulled']
    pullingSpaceArray = np.unique(list(itertools.permutations(pullingAgents)), axis=0)
    pullingSpace = [tuple(pullingPair) for pullingPair in pullingSpaceArray.tolist()]
    actionHypo = list(itertools.product(actionSpace, actionSpace, actionSpace))
    iterables = [chasingSpace, pullingSpace, actionHypo]
    inferenceIndex = pd.MultiIndex.from_product(iterables, names=['mind', 'physics', 'action'])

    drawInferenceResult = DrawInferenceResult(inferenceIndex, gridPixelSize, initializeGame, drawGrid, drawCirclesAndLines,
                 colorChasingPoints, adjustPullingLineWidth)


    thresholdPosterior = 1
    mindPhysicsName = ['mind', 'physics']
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior, mindPhysicsName, inferenceIndex)

    mindPhysicsActionName = ['mind', 'physics', 'action']
    inferOneStep = InferOneStep(inferenceIndex, mindPhysicsActionName, getMindsPhysicsActionsJointLikelihood)

    trajectory = [[(6, 2), (9, 2), (5, 4)], [(7, 3), (10, 2), (6, 3)], [(6, 2), (10, 2), (7, 2)], [(8, 2), (10, 2), (6, 1)], [(8, 2), (10, 2), (7, 1)], [(8, 2), (10, 3), (7, 1)], [(9, 1), (10, 3), (8, 2)], [(8, 2), (10, 3), (8, 2)], [(8, 3), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (7, 2)], [(8, 3), (10, 3), (8, 1)], [(9, 2), (10, 3), (8, 1)], [(10, 1), (10, 4), (7, 2)], [(9, 2), (10, 5), (9, 2)], [(9, 3), (9, 5), (9, 3)], [(9, 4), (9, 6), (10, 3)], [(10, 5), (9, 7), (10, 3)], [(9, 4), (10, 7), (10, 4)], [(10, 5), (10, 7), (10, 4)], [(10, 5), (10, 8), (9, 5)], [(9, 6), (10, 9), (10, 4)], [(10, 7), (10, 9), (8, 4)], [(9, 8), (10, 10), (9, 3)], [(10, 7), (10, 10), (10, 4)], [(10, 7), (10, 10), (9, 5)], [(10, 7), (10, 10), (8, 6)], [(9, 8), (10, 10), (10, 6)], [(9, 8), (10, 10), (10, 6)], [(10, 9), (10, 9), (9, 7)]]
    observe = Observe(trajectory)

    inferDiscreteChasingAndDrawDemo = InferDiscreteChasingAndDrawDemo(isInferenceTerminal, observe, inferOneStep,
                 drawInferenceResult, saveImage)

    mindsPhysicsPrior = [1/ len(inferenceIndex)] * len(inferenceIndex)
    posterior = inferDiscreteChasingAndDrawDemo(mindsPhysicsPrior)
    print(max(posterior)* 4*4*4)

if __name__ == '__main__':
    main()
