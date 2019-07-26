import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from exec.trajectoriesSaveLoad import loadFromPickle
from exec.generateVideos.chasingVisualization import DrawBackground, DrawState, ChaseTrialWithTraj, Observe
from exec.generateVideos.trajectory import ScaleTrajectory, AdjustDfFPStoTraj
from visualize.initialization import initializeScreen

import pandas as pd
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    dirName = os.path.dirname(__file__)
    dataIndex = 5
    dataPath = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively', 'replayBufferStartWithTrainedModel', 'evaluationTrajectories20kTrainSteps', 'iteration=19999_maxRunningSteps=20_numTrials=500_policyName=NNPolicy_sampleIndex=493_trainBufferSize=2000_trainLearningRate=0.0001_trainMiniBatchSize=256_trainNumSimulations=200_trainNumTrajectoriesPerIteration=1.pickle')
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
    obstacle1Pos = [377.5, 270, 45, 60]
    obstacle2Pos = [377.5, 470, 45, 60]
    allObstaclePos = [obstacle1Pos, obstacle2Pos]
    screenColor = THECOLORS['black']
    lineColor = THECOLORS['white']

    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    circleSize = 10
    positionIndex = [0, 1]
    drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

    colorSpace = [THECOLORS['green'], THECOLORS['red']]

    FPS = 60
    chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True, imageFolderName='2ObjectsDemo' + str(dataIndex))

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
    chaseTrial(positionListToDraw)


if __name__ == '__main__':
    main()
