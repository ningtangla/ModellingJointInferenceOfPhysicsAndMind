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
    maxRunningSteps = 30
    numSimulations = 200
    killzoneRadius = 2
    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    numTrials=7
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius,'numTrials':numTrials}
    # trajectoryDirectory = os.path.join(dirName, '..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentPhysicsWithObstacle', 'trajectories')
    trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'multiMCTSAgentPhysicsWithObstacle','evaluateMCTSSimulation', 'trajectories')
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = []
    # fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    para = {'numSimulations':numSimulations }
    allTrajectories = loadTrajectories(para)
    print(allTrajectories)
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
            obstacle1Pos = [390.5, 289.8, 19, 106]
            obstacle2Pos = [390.5, 403.8, 19, 106]
            allObstaclePos = [obstacle1Pos, obstacle2Pos]
            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']

            drawBackground = DrawBackgroundWithObstacles(screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth)
            circleSize = 4
            positionIndex = [0, 1]
            drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

            colorSpace = [THECOLORS['green'], THECOLORS['red']]

            FPS = 60

            chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=False,)

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
            chaseTrial(2,positionListToDraw, '2ObjectsObstaclesNNGuideMCTS_selfIteration=6500_otherIteration=6500/' + str(dataIndex))


if __name__ == '__main__':
    main()
