from pygame.color import THECOLORS
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualize.continuousVisualization import ScaleTrajectory, AdjustDfFPStoTraj,\
    DrawBackground, DrawState, ChaseTrialWithTraj
from visualize.initialization import initializeScreen

def getTrajectoryFromDf(trajectoryDf, positionIndex):
    xIndex, yIndex = positionIndex
    unorderedTraj = [[value[xIndex], value[yIndex]] for key, value in trajectoryDf.iterrows()]
    trajectory = [[unorderedTraj[i], unorderedTraj[i+1]] for i in range(0, len(unorderedTraj), 2)]
    return trajectory


def main():
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

    colorSpace = [THECOLORS['green'], THECOLORS['red']]

    FPS = 60
    chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True, imageFolderName='demo1')

    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [210, 590]
    scaledYRange = [210, 590]
    scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

    oldFPS = 5
    adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

    getTrajectory = lambda rawTrajectory: scaleTrajectory(adjustFPS(rawTrajectory))

    parentPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dataPath = os.path.abspath(os.path.join(parentPath, 'trainedData'))

    rawTrajectoryDf = pd.read_pickle(os.path.join(os.path.join(dataPath, 'df1.pickle')))
    rawTrajectory = getTrajectoryFromDf(rawTrajectoryDf, positionIndex)

    trajectory = getTrajectory(rawTrajectory)
    chaseTrial(trajectory)

if __name__ == '__main__':
    main()

