import sys
import pandas as pd

sys.path.append('../src/constrainedChasingEscapingEnv')
sys.path.append('../visualize')
sys.path.append('../src')


from envDiscreteGrid import *
from wrapperFunctions import *
from policies import *
from analyticGeometryFunctions import computeAngleBetweenVectors

from discreteGridVisualization import *

from play import MultiAgentSampleTrajectory

pd.set_option('display.max_columns', 50)


def main():

    rationalityParam = 0.9
    actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]
    gridSize = (10,10)
    iterationNumber = 50
    agentNames = ["Wolf", "Sheep", "Master"]


    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2

    getWolfProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)
    getSheepProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)

    lowerGridBound = 1
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerGridBound)

    wolfID = 0
    sheepID = 1
    masterID = 2
    positionIndex = [0,1]

    getPredatorPos = GetAgentPosFromState(wolfID, positionIndex)
    getPreyPos = GetAgentPosFromState(sheepID, positionIndex)
    getMasterPos = GetAgentPosFromState(masterID, positionIndex)

    getWolfPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, getWolfProperAction, getPredatorPos, getPreyPos)
    getSheepPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, getSheepProperAction, getPredatorPos, getPreyPos)
    getMasterPolicy = RandomPolicy(actionSpace)

    allAgentPolicyFunction = pd.Series(data = [getWolfPolicy, getSheepPolicy, getMasterPolicy], index = [wolfID, sheepID, masterID]).sort_index().tolist()


    policy = lambda state: [getAction(state) for getAction in allAgentPolicyFunction]


    agentCount = 3
    reset = Reset(gridSize, lowerGridBound, agentCount)

    adjustingParam = 3
    getPullingForceValue = GetPullingForceValue(adjustingParam, roundNumber)
    samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors, actionSpace, lowerBoundAngle, upperBoundAngle)


    pulledAgentID = 0
    noPullAgentID = 1
    pullingAgentID = 2

    getPulledAgentPos = GetAgentPosFromState(pulledAgentID, positionIndex)
    getNoPullAgentPos = GetAgentPosFromState(noPullAgentID, positionIndex)
    getPullingAgentPos = GetAgentPosFromState(pullingAgentID, positionIndex)

    getPulledAgentForce = GetPulledAgentForce(getPullingAgentPos, getPulledAgentPos, samplePulledForceDirection, getPullingForceValue)
    getNoPullingForce= GetNoForceAgentForce(getPullingAgentPos, getPulledAgentPos)
    getPullingAgentForce = GetPulledAgentForce(getPulledAgentPos, getPullingAgentPos, samplePulledForceDirection, getPullingForceValue)

    transitPulledAgent = TransitAgent(stayWithinBoundary, getPulledAgentForce, getPulledAgentPos)
    transitNoPullAgent = TransitAgent(stayWithinBoundary, getNoPullingForce, getNoPullAgentPos)
    transitPullingAgent = TransitAgent(stayWithinBoundary, getPullingAgentForce, getPullingAgentPos)
    allAgentTransitionFunction = pd.Series(data = [transitPulledAgent, transitNoPullAgent, transitPullingAgent], index = [pulledAgentID, noPullAgentID, pullingAgentID]).sort_index().tolist()

    transition = lambda allAgentActions, state: [transitAgent(action, state) for transitAgent, action in zip(allAgentTransitionFunction, allAgentActions)]


    isTerminal = IsTerminal(getPredatorPos, getPreyPos)

    getMultiAgentSampleTrajectory = MultiAgentSampleTrajectory(agentNames, iterationNumber, isTerminal, reset)

    locationDf = getMultiAgentSampleTrajectory(policy, transition)
    print(locationDf)



    BLACK = (  0,   0,   0)
    WHITE = (255, 255, 255)
    BLUE =  (  0,   0, 255)
    PINK = ( 250,   0, 255)
    GREEN = (0, 255, 0)


    screenWidth = 800
    screenHeight = 800
    gridNumberX, gridNumberY = gridSize
    gridPixelSize = min(screenHeight// gridNumberX, screenWidth// gridNumberY)


    pointExtendTime = 100
    FPS = 60
    colorList = [BLUE, PINK, GREEN]
    pointWidth = 10
    modificationRatio = 3


    modifyOverlappingPoints = ModifyOverlappingPoints(gridPixelSize, modificationRatio, checkDuplicates)
    drawCircles = DrawCircles(pointExtendTime, FPS, colorList , pointWidth, modifyOverlappingPoints)

    caption= "Game"
    initializeGame = InitializeGame(screenWidth, screenHeight, caption)

    gridColor = BLACK
    gridLineWidth = 3
    backgroundColor= WHITE
    drawGrid = DrawGrid(gridSize, gridPixelSize, backgroundColor, gridColor, gridLineWidth)

    drawPointsFromLocationDfandSaveImage =  DrawPointsFromLocationDfAndSaveImage(initializeGame, drawGrid, drawCircles, gridPixelSize)
    drawPointsFromLocationDfandSaveImage(locationDf, iterationNumber, saveImage = False)



if __name__ == '__main__':
    main()

