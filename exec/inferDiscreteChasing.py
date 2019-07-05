import sys

sys.path.append('../src/inferDiscreteGridChasing')
sys.path.append('../src/constrainedChasingEscapingEnv')

from forceLikelihood import *
from heatSeekingLikelihood import *
from analyticGeometryFunctions import computeAngleBetweenVectors
from inference import *
from wrapperFunctions import *
from wrappers import *
from discreteGridInferenceVisualization import *

pd.set_option('display.max_columns', 50)

def main():
    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2
    actHeatSeeking = ActHeatSeeking(actionSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors)
    positionIndex = [0, 1]
    getAgentPosition = lambda agentID: GetAgentPosFromState(agentID, positionIndex)
    rationalityParam = 0.9
    getHeatSeekingActionLikelihood = lambda getWolfPos, getSheepPos: HeatSeekingActionLikelihood(rationalityParam, actHeatSeeking, getWolfPos, getSheepPos)
    getWolfActionProb = GetWolfActionProb(getAgentPosition, getHeatSeekingActionLikelihood)
    getSheepActionProb = GetSheepActionProb(getAgentPosition, getHeatSeekingActionLikelihood)
    getRandomActionLikelihood = GetRandomActionLikelihood(actionSpace)
    getMasterActionProb = GetMasterActionProb(getRandomActionLikelihood)
    getAgentsActionProb = [getWolfActionProb, getSheepActionProb, getMasterActionProb]
    getPolicyLikelihood = lambda chasingIndices, state, allAgentsAction: np.product([getActionProb(chasingIndices, state, allAgentsAction) for getActionProb in getAgentsActionProb] )

    forceSpace = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
    getPulledAgentForceLikelihood = PulledForceDirectionLikelihood(forceSpace, lowerBoundAngle, upperBoundAngle, computeAngleBetweenVectors) # change or not
    getPulledAgentForceProb = GetPulledAgentForceProb(getAgentPosition, getPulledAgentForceLikelihood)
    gridSize = (10, 10)
    lowerBoundary = 1
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerBoundary)
    getTransitionLikelihood = GetTransitionLikelihood(getPulledAgentForceProb, getNoPullAgentForceProb, stayWithinBoundary)

    inferOneStepDiscreteChasing = InferOneStepDiscreteChasing(getPolicyLikelihood, getTransitionLikelihood)

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

    drawInferenceResult = DrawInferenceResult(gridPixelSize, initializeGame, drawGrid, drawCirclesAndLines,
                                              colorChasingPoints, adjustPullingLineWidth)

    thresholdPosterior = 1
    isInferenceTerminal = IsInferenceTerminal(thresholdPosterior)

    chasingAgents = ['wolf', 'sheep', 'master']
    pullingAgents = ['pulled', 'noPull', 'pulled']
    inferDiscreteChasingAndDrawDemo = InferDiscreteChasingAndDrawDemo(chasingAgents, pullingAgents, actionSpace, forceSpace,
                                                createIndex, isInferenceTerminal, inferOneStepDiscreteChasing, drawInferenceResult)

    # trajectory = [[(5, 3), (8, 8), (6, 4)], [(5, 5), (8, 9), (5, 3)],
    #               [(6, 4), (8, 10), (4, 4)], [(6, 4), (7, 10), (6, 4)],
    #               [(5, 4), (6, 10), (7, 4)], [(6, 5), (5, 10), (5, 4)],
    #               [(5, 6), (5, 10), (5, 4)], [(5, 6), (5, 10), (4, 5)],
    #               [(5, 6), (5, 10), (4, 5)], [(5, 6), (5, 10), (4, 5)],
    #               [(4, 7), (5, 10), (5, 4)], [(5, 6), (5, 10), (5, 6)],
    #               [(5, 7), (5, 10), (5, 7)], [(5, 8), (5, 10), (5, 8)],
    #               [(5, 9), (5, 10), (5, 7)], [(5, 9), (5, 10), (5, 9)],
    #               [(5, 10), (5, 10), (6, 9)]] # 17 states

    trajectory = [[(7, 7), (7, 4), (4, 6)], [(6, 6), (7, 3), (4, 6)], [(5, 5), (8, 3), (4, 6)], [(4, 4), (7, 3), (4, 6)],
     [(4, 6), (7, 2), (4, 6)], [(5, 6), (7, 1), (5, 6)], [(6, 6), (7, 2), (5, 5)], [(6, 4), (7, 1), (5, 5)],
     [(5, 3), (7, 1), (7, 5)], [(6, 4), (7, 1), (7, 5)], [(7, 3), (7, 2), (5, 5)], [(6, 2), (7, 1), (6, 6)],
     [(6, 2), (8, 1), (5, 5)], [(6, 2), (7, 1), (5, 5)], [(6, 4), (7, 1), (5, 3)], [(7, 3), (8, 1), (5, 5)],
     [(6, 2), (9, 1), (5, 5)], [(7, 3), (8, 1), (5, 3)], [(7, 3), (8, 1), (6, 4)], [(8, 4), (9, 1), (6, 4)],
     [(6, 4), (10, 1), (7, 3)], [(7, 3), (10, 1), (6, 2)], [(6, 2), (10, 1), (7, 3)], [(6, 2), (10, 1), (7, 3)],
     [(7, 1), (10, 1), (5, 3)], [(7, 1), (10, 1), (6, 4)], [(7, 1), (10, 1), (7, 5)], [(8, 2), (10, 1), (8, 4)],
     [(8, 4), (10, 2), (7, 3)], [(6, 4), (10, 2), (9, 3)], [(6, 2), (10, 1), (9, 3)], [(7, 1), (10, 1), (7, 3)],
     [(8, 2), (10, 1), (7, 1)], [(9, 1), (10, 1), (6, 2)], [(10, 2), (10, 1), (7, 1)], [(9, 1), (10, 1), (8, 1)],
     [(9, 1), (10, 1), (9, 1)], [(10, 1), (10, 1), (10, 1)]] # 38 states

    inferenceDf = inferDiscreteChasingAndDrawDemo(trajectory)

    inferenceDf.to_csv("chasingInference.csv")

if __name__ == '__main__':
    main()
