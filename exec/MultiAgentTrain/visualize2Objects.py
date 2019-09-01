import sys
import os
import numpy as np
import pandas as pd
from pygame.color import THECOLORS
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from exec.trajectoriesSaveLoad import loadFromPickle
from src.inferChasing.inference import Observe
from visualize.continuousVisualization import DrawBackground
   

from exec.generateExpDemo.trajectory import ScaleTrajectory, AdjustDfFPStoTraj

from visualize.initialization import initializeScreen

from exec.generateExpDemo.chasingVisualization import  DrawState, ChaseTrialWithTraj, DrawStateWithRope, ChaseTrialWithRopeTraj

from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, \
    loadFromPickle, saveToPickle, LoadTrajectories, GetSavePath, conditionDfFromParametersDict,GetAgentCoordinateFromTrajectoryAndStateDf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    dirName = os.path.dirname(__file__)
    dataIndex = 5
    dataPath = os.path.join(dirName, '..', '..', 'data', 'multiAgentTrain','multiMCTSAgent','evaluateTrajectories','killzoneRadius=2_maxRunningSteps=20_numSimulations=200_otherIteration=18000_sampleIndex=(0,46)_selfId=0_selfIteration=0.pickle')
    trajectory = list(loadFromPickle(dataPath)[42])
    # del trajectory[-1]
    print(trajectory)
    stateIndex = 0

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

    stateIndex = 0
    getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][0])[0]))
    getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
    getAllLevelValuesRange = {'timeStep': getRangeTrajectoryLength, 'agentId': getRangeNumAgentsFromTrajectory}

    getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 0)
    getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 1)
    extractColumnValues = {'xPos': getAgentPosXCoord, 'yPos': getAgentPosYCoord}

    convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(getAllLevelValuesRange, conditionDfFromParametersDict, extractColumnValues)
    trajectoryDF = convertTrajectoryToStateDf(trajectory)

    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    circleSize = 10
    positionIndex = [0, 1]
    numOfAgent = 2
    drawState = DrawState(screen, circleSize, numOfAgent, positionIndex, drawBackground)

    circleColorList = [THECOLORS['green'], THECOLORS['red']]

    FPS = 60

    saveImageDir = os.path.join(dirName, '..', '..', 'data', 'multiAgentTrain','demo','twoAgents')
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    chaseTrial = ChaseTrialWithTraj(FPS, circleColorList, drawState, saveImage=True, saveImageDir=saveImageDir)

    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [200, 600]
    scaledYRange = [200, 600]
    scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

    oldFPS = 5
    adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

    getTrajectory = lambda rawTrajectory: scaleTrajectory(adjustFPS(rawTrajectory))
    trajectoryToDraw = getTrajectory(trajectoryDF)
    chaseTrial(trajectoryToDraw)


if __name__ == '__main__':
    main()
