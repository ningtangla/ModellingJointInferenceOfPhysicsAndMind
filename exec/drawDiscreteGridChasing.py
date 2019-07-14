import sys
import pandas as pd
import numpy as np

sys.path.append('../src/constrainedChasingEscapingEnv')
sys.path.append('../visualize')
sys.path.append('../src')

from envDiscreteGrid import Reset, StayWithinBoundary, GetPullingForceValue, \
    GetPulledAgentForce, SamplePulledForceDirection, GetAgentsForce, Transition, IsTerminal

from state import GetAgentPosFromState
from policies import ActHeatSeeking, HeatSeekingDiscreteStochasticPolicy, RandomPolicy
from analyticGeometryFunctions import computeAngleBetweenVectors
from discreteGridVisualization import DrawGrid, checkDuplicates, \
    ModifyOverlappingPoints, DrawCircles, DrawPointsAndSaveImage

from episode import MultiAgentSampleTrajectory
from initialization import initializeScreen


def main():

    rationalityParam = 0.9
    actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1)]

    gridSize = (10,10)
    iterationNumber = 50
    agentNames = ["Wolf", "Sheep", "Master"]

    lowerBoundAngle = 0
    upperBoundAngle = np.pi / 2

    getWolfProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)
    getSheepProperAction = ActHeatSeeking(actionSpace, computeAngleBetweenVectors, lowerBoundAngle, upperBoundAngle)


    wolfID = 0
    sheepID = 1
    masterID = 2
    positionIndex = [0, 1] # [2, 0, 1] state = [sheepState, masterState, wolfState]

    getWolfPos = GetAgentPosFromState(wolfID, positionIndex)
    getSheepPos = GetAgentPosFromState(sheepID, positionIndex)

    getWolfPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, getWolfProperAction, getWolfPos, getSheepPos)
    getSheepPolicy = HeatSeekingDiscreteStochasticPolicy(rationalityParam, getSheepProperAction, getWolfPos, getSheepPos)
    getMasterPolicy = RandomPolicy(actionSpace)

    unorderedPolicy = [getWolfPolicy, getSheepPolicy, getMasterPolicy]
    agentID = [wolfID, sheepID, masterID]

    rearrangeList = lambda unorderedList, order: list(np.array(unorderedList)[np.array(order).argsort()])
    allAgentPolicy = rearrangeList(unorderedPolicy, agentID) # sheepPolicy, masterPolicy, wolfPolicy

    policy = lambda state: [getAction(state) for getAction in allAgentPolicy] # result of policy function is [sheepAct, masterAct, wolfAct]

    pulledAgentID = 0
    noPullAgentID = 1
    pullingAgentID = 2

    getPulledAgentPos = GetAgentPosFromState(pulledAgentID, positionIndex)
    getPullingAgentPos = GetAgentPosFromState(pullingAgentID, positionIndex)
    lowerGridBound = 1
    agentCount = 3
    reset = Reset(gridSize, lowerGridBound, agentCount)
    stayWithinBoundary = StayWithinBoundary(gridSize, lowerGridBound)

    distanceForceRatio = 20
    getPullingForceValue = GetPullingForceValue(distanceForceRatio)
    samplePulledForceDirection = SamplePulledForceDirection(computeAngleBetweenVectors, actionSpace, lowerBoundAngle, upperBoundAngle)

    getPulledAgentForce = GetPulledAgentForce(getPullingAgentPos, getPulledAgentPos, samplePulledForceDirection, getPullingForceValue)
    getAgentsForce = GetAgentsForce(getPulledAgentForce, pulledAgentID, noPullAgentID, pullingAgentID)
    transition = Transition(stayWithinBoundary, getAgentsForce)
    isTerminal = IsTerminal(getWolfPos, getSheepPos)

    getMultiAgentSampleTrajectory = MultiAgentSampleTrajectory(agentNames, iterationNumber, isTerminal, reset)
    trajectory = getMultiAgentSampleTrajectory(policy, transition)

    print(trajectory)
    print(len(trajectory))

    BLACK = (  0,   0,   0)
    colorList = [BLACK ,BLACK ,BLACK]

    fullScreen = False
    screenWidth = 800
    screenHeight = 800
    screen = initializeScreen(fullScreen, screenWidth, screenHeight)

    gridNumberX, gridNumberY = gridSize
    gridPixelSize = min(screenHeight// gridNumberX, screenWidth// gridNumberY)
    drawGrid = DrawGrid(screen, gridSize, gridPixelSize)

    modificationRatio = 3
    modifyOverlappingPoints = ModifyOverlappingPoints(gridPixelSize, checkDuplicates, modificationRatio)

    drawCircles = DrawCircles(colorList, modifyOverlappingPoints)

    drawPointsFromLocationDfandSaveImage = DrawPointsAndSaveImage(drawGrid, drawCircles, gridPixelSize)
    drawPointsFromLocationDfandSaveImage(trajectory, iterationNumber, saveImage = False)


if __name__ == '__main__':
    main()

