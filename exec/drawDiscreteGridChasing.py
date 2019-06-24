import sys
import pandas as pd 
import numpy as np 
from random import randint

sys.path.append('../src/sheepWolf')
sys.path.append('../visualize')


from envDiscreteGrid import *
from discreteGridWrapperFunctions import *
from discreteGridPolicyFunctions import *
from calculateAngleFunction import *

from discreteGridVisualization import *

pd.set_option('display.max_columns', 50)


def main():

    rationalityParam = 0.9
    actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1), (0, 0)]
    gridSize = (10,10)
    iterationNumber = 50
    agentNames = ["Wolf", "Sheep", "Master"]


    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2

    getWolfProperAction = ActHeatSeeking(actionSpace, calculateAngle, lowerBoundAngle, upperBoundAngle)
    getSheepProperAction = ActHeatSeeking(actionSpace, calculateAngle, lowerBoundAngle, upperBoundAngle)
    
    lowerGridBound = 1
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerGridBound)

    wolfID = 0
    sheepID = 1
    masterID = 2
    positionIndex = [0,1]

    locateWolf = LocateAgent(wolfID, positionIndex)
    locateSheep = LocateAgent(sheepID, positionIndex)
    locateMaster = LocateAgent(masterID, positionIndex)

    getWolfPolicy = HeatSeekingPolicy(rationalityParam, getWolfProperAction, locateWolf, locateSheep)
    getSheepPolicy = HeatSeekingPolicy(rationalityParam, getSheepProperAction, locateWolf,locateSheep)
    getMasterPolicy = RandomActionPolicy(actionSpace)

    allAgentPolicyFunction = [getWolfPolicy, getSheepPolicy, getMasterPolicy]
    policy = lambda state: [getAction(state) for getAction in allAgentPolicyFunction]



    agentCount = 3
    reset = Reset(gridSize, lowerGridBound, agentCount)

    adjustingParam = 3 
    getPullingForceValue = GetPullingForceValue(adjustingParam, roundNumber)
    
    samplePulledForceDirection = SamplePulledForceDirection(calculateAngle, actionSpace, lowerBoundAngle, upperBoundAngle)

    getAgentsForceAction = GetAgentsForceAction(locateMaster, locateWolf, samplePulledForceDirection, getPullingForceValue)

    transition = Transition(stayWithinBoundary, getAgentsForceAction)
  

    isTerminal = IsTerminal(locateWolf, locateSheep)

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
    colorList = [BLUE, PINK, GREEN] # wolf, sheep, master
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
    drawPointsFromLocationDfandSaveImage(locationDf, iterationNumber, saveImage = True)



if __name__ == '__main__':
    main()