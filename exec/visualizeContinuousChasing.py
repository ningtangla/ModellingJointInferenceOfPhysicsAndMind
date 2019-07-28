import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from src.inferChasing.inference import Observe
from exec.trajectoriesSaveLoad import loadFromPickle
from visualize.continuousVisualization import ScaleTrajectory, AdjustDfFPStoTraj,\
    DrawBackground, DrawState, ChaseTrialWithTraj
from visualize.initialization import initializeScreen

import pandas as pd
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    dirName = os.path.dirname(__file__)
    dataIndex = 2
    dataPath = os.path.join(dirName, '..', 'trainedData', 'leasedTraj'+ str(dataIndex) + '.pickle')
    trajectory = loadFromPickle(dataPath)

    stateIndex = 0
    observe = Observe(stateIndex, trajectory)

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

    numberOfAgents = 3
    chasingColors = [THECOLORS['green'], THECOLORS['red'], THECOLORS['blue']]
    colorSpace = chasingColors[: numberOfAgents]

    FPS = 60
    chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True)

    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [210, 590]
    scaledYRange = [210, 590]
    scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

    oldFPS = 5
    adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

    getTrajectory = lambda rawTrajectory: scaleTrajectory(adjustFPS(rawTrajectory))
    positionList = [observe(index) for index in range(len(trajectory))]
    positionListToDraw = getTrajectory(positionList)

    currentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
    imageFolderName = 'leasedObjectsDemo' + str(dataIndex)
    saveImageDir = os.path.join(os.path.join(parentDir, 'demo'), imageFolderName)
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)

    chaseTrial(positionListToDraw, saveImageDir)


if __name__ == '__main__':
    main()
