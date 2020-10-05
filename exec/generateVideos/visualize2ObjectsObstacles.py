import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from exec.trajectoriesSaveLoad import loadFromPickle,GetSavePath,LoadTrajectories
from src.inferChasing.inference import Observe
from visualize.continuousVisualization import DrawBackgroundWithObstacles, DrawState, ChaseTrialWithTraj, \
    ScaleTrajectory, AdjustDfFPStoTraj
from visualize.initialization import initializeScreen

import pandas as pd
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    dirName = os.path.dirname(__file__)
    # dataPath = os.path.join(dirName, '..', '..', 'data', 'multiAgentTrain','multiMCTSAgentObstacle','evaluateTrajectories','killzoneRadius=2_maxRunningSteps=30_numSimulations=200_otherIteration=6000_sampleIndex=(15,16)_selfId=0_selfIteration=6000.pickle')
    # trajectory = loadFromPickle(dataPath)[0]
    # trajectorygg = trajectory.copy()
    # del trajectory[-1]
    # print(trajectorygg.pop())

    trajectoryFixedParameters = {'killzoneRadius':2,'maxRunningSteps':30,'numSimulations':200}
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'multiAgentTrain','multiMCTSAgentObstacle','demoTrajectoriesNNGuideMCTS')

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = ['sampleIndex']
    # fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    para = {'selfIteration':6500,'otherIteration':6500,'selfId':0  }
    allTrajectories = loadTrajectories(para)

    for dataIndex in range(len(allTrajectories)):
        trajectory = allTrajectories[dataIndex]
        del trajectory[-1]
        if len(trajectory) != 0:
            # print(trajectory[0])
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
            obstacle1Pos = [380, 273, 40, 54]
            obstacle2Pos = [380, 473, 40, 54]
            allObstaclePos = [obstacle1Pos, obstacle2Pos]
            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']

            drawBackground = DrawBackgroundWithObstacles(screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth)
            circleSize = 10
            positionIndex = [0, 1]
            drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

            colorSpace = [THECOLORS['green'], THECOLORS['red']]

            FPS = 60

            chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True, imageFolderName='2ObjectsObstaclesNNGuideMCTS_selfIteration=6500_otherIteration=6500/' + str(dataIndex))

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
