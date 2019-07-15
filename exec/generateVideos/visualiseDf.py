from pygame.color import THECOLORS
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from exec.generateVideos.trajectory import ScaleTrajectory, AdjustDfFPStoTraj
from exec.generateVideos.chasingVisualization import InitializeScreen, DrawBackground, DrawState, ChaseTrialWithTraj

def main():
    screenWidth = 800
    screenHeight = 800
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

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
    chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True, imageFolderName='videosNumSim50/9')

    rawXRange = [-10, 10]
    rawYRange = [-10, 10]
    scaledXRange = [200, 600]
    scaledYRange = [200, 600]
    scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

    oldFPS = 5
    adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

    getTrajectory = lambda trajectoryDf: scaleTrajectory(adjustFPS(trajectoryDf))

    parentPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dataPath = os.path.abspath(os.path.join(parentPath, 'trainedData'))
    DIRNAME = os.path.dirname(__file__)
    trajectoryDf = pd.read_pickle(os.path.join(DIRNAME, '..', '..', 'data',
                                       'evaluateNumTerminalTrajectoriesWolfChaseSheepMCTSRolloutMujoco', 'videosNumSim50',
                                               'sampleIndex4.pickle'))
    #pd.read_pickle(os.path.join(os.path.join(dataPath, 'df.pickle')))
    trajectory = getTrajectory(trajectoryDf)
    chaseTrial(trajectory)


if __name__ == '__main__':
    main()
