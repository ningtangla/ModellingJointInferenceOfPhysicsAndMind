import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from exec.trajectoriesSaveLoad import loadFromPickle,GetSavePath,LoadTrajectories
from src.inferChasing.inference import Observe
from visualize.continuousVisualization import DrawBackgroundWithObstacles,ScaleTrajectory, AdjustDfFPStoTraj
from visualize.initialization import initializeScreen

import pandas as pd
import numpy as np
from pygame.color import THECOLORS
import pygame as pg
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
    numTrials=6
    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius,'maxRolloutSteps':30,'agentId':1}
    # trajectoryDirectory = os.path.join(dirName, '..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentPhysicsWithObstacle', 'trajectories')
    # trajectoryDirectory = os.path.join(dirName, '..', '..', 'data', 'multiMCTSAgentPhysicsWithObstacle','evaluateMCTSSimulation', 'trajectories')
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,'killzoneRadius': killzoneRadius}
    trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentObstacle', 'trajectories')
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    fuzzySearchParameterNames = []
    # fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    # para = {'numSimulations':numSimulations }
    iterationIndex=1895
    para = {'iterationIndex':iterationIndex }
    allTrajectories = loadTrajectories(para)
    print(len(allTrajectories))
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
            rescaleObstacle1Pos = [390.5, 328.5, 19, 66.5]
            rescaleObstacle2Pos = [390.5, 404.75, 19, 66.5]
            allObstaclePos = [rescaleObstacle1Pos, rescaleObstacle2Pos]
            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']

            drawBackground = DrawBackgroundWithObstacles(screen, screenColor, xBoundary, yBoundary, allObstaclePos, lineColor, lineWidth)
            circleSize = 8#4
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
            demoDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentObstacle', 'demo')

            if not os.path.exists(demoDirectory):
                os.makedirs(demoDirectory)
            chaseTrial(2,positionListToDraw, os.path.join(demoDirectory,str(iterationIndex)))

class DrawState:
    def __init__(self, screen, circleSize, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround

    def __call__(self,numOfAgent, state, circleColorList):
        self.drawBackGround()
        
        for agentIndex in range(numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        return self.screen
class ChaseTrialWithTraj:
    def __init__(self, fps, colorSpace, drawState, saveImage):
        self.fps = fps
        self.colorSpace = colorSpace
        self.drawState = drawState
        self.saveImage = saveImage
    def __call__(self, numOfAgents, trajectoryData, imagePath):
        if not os.path.exists(imagePath):
            os.makedirs(imagePath)
        fpsClock = pg.time.Clock()
        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(self.fps)
            screen = self.drawState(numOfAgents, state, self.colorSpace)
            if self.saveImage == True:
                filename='Obstacle'+format(timeStep, '04') + ".png"
                pg.image.save(screen, os.path.join(imagePath,filename ))
        return

if __name__ == '__main__':
    main()
