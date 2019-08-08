from pygame.color import THECOLORS
import os
import pandas as pd
import sys
import numpy as np
import itertools as it
from collections import OrderedDict
DIRNAME = os.path.dirname(__file__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf, \
    loadFromPickle, saveToPickle, LoadTrajectories, GetSavePath,conditionDfFromParametersDict
from exec.generateVideosForLeashedDemo.trajectory import ScaleTrajectory, AdjustDfFPStoTraj
from exec.generateVideosForLeashedDemo.chasingVisualization import InitializeScreen, DrawBackground, DrawState, ChaseTrialWithTraj

def getFileName(parameters,fixedParameters):
    allParameters = dict(list(parameters.items()) + list(fixedParameters.items()))
    sortedParameters = sorted(allParameters.items())
    nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]
    fileName = '_'.join(nameValueStringPairs)

    return fileName

def main():
    manipulatedVariables = OrderedDict()

    # manipulatedVariables['draggerMass'] = [8, 10, 12]
    # manipulatedVariables['maxTendonLength'] = [0.6]
    # manipulatedVariables['predatorMass'] = [10]
    # manipulatedVariables['predatorPower'] = [1, 1.3, 1.6]
    # manipulatedVariables['tendonDamping'] =[0.7]
    # manipulatedVariables['tendonStiffness'] = [10]

    # manipulatedVariables['agentId'] = [1]
    # manipulatedVariables['maxRunningSteps'] = [25]
    # manipulatedVariables['numSimulations'] = [200]
    # manipulatedVariables['miniBatchSize'] = [256]#[64, 128, 256, 512]
    # manipulatedVariables['learningRate'] =  [1e-4]#[1e-2, 1e-3, 1e-4, 1e-5]
    # manipulatedVariables['depth'] = [4]#[2,4, 6, 8]
    # manipulatedVariables['trainSteps'] = [20000]#list(range(0,100001, 20000))

    manipulatedVariables['safeBound'] = [1.5]
    manipulatedVariables['preyPowerRatio'] =[0.7]
    manipulatedVariables['wallPunishRatio'] = [0.6]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditionParametersAll = [dict(list(i)) for i in productedValues]

    trajectoryFixedParameters = {}
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'searchToWallHerustic', 'mctsSheep')

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    # fuzzySearchParameterNames = ['sampleIndex']
    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle,fuzzySearchParameterNames)

    agentId = 1
    stateIndex = 0
    getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][stateIndex])[0]))
    getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
    getAllLevelValuesRange = {'timeStep': getRangeTrajectoryLength, 'agentId': getRangeNumAgentsFromTrajectory}

    getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 0)
    getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 1)
    getAgentVelXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 4)
    getAgentVelYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 5)
    extractColumnValues = {'xPos': getAgentPosXCoord, 'yPos': getAgentPosYCoord, 'xVel': getAgentVelXCoord,
                           'yVel': getAgentVelYCoord}
    convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(getAllLevelValuesRange, conditionDfFromParametersDict,extractColumnValues)


#### convert traj pickle to df
    for conditionParameters in conditionParametersAll:
        trajectories = loadTrajectories(conditionParameters)
        numTrajectories = len(trajectories)

        numTrajectoryChoose = min(numTrajectories, 10)
        selectedTrajectories = trajectories[0:numTrajectoryChoose]
        selectedDf = [convertTrajectoryToStateDf(trajectory) for trajectory in selectedTrajectories]

        dataFileName = getFileName(conditionParameters,trajectoryFixedParameters)
        imageSavePath = os.path.join(trajectoryDirectory, dataFileName)
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)

        [saveToPickle(df, os.path.join(imageSavePath, 'sampleIndex={}.pickle'.format(sampleIndex))) for df, sampleIndex in zip(selectedDf, range(numTrajectories))]

### generate demo image
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

        numOfAgent = 3
        drawState = DrawState(screen, circleSize, positionIndex, drawBackground,numOfAgent)
        colorSpace = [THECOLORS['green'], THECOLORS['red'], THECOLORS['blue']]

        for index in range(numTrajectoryChoose):
            imageFolderName = "{}".format(index)
            saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))

            FPS = 60
            chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True, saveImageDir=saveImageDir)

            rawXRange = [-10, 10]
            rawYRange = [-10, 10]
            scaledXRange = [200, 600]
            scaledYRange = [200, 600]
            scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

            oldFPS = 5
            adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

            getTrajectory = lambda trajectoryDf: scaleTrajectory(adjustFPS(trajectoryDf))
            trajectoryDf = pd.read_pickle(os.path.join(imageSavePath, 'sampleIndex={}.pickle'.format(index)))
            trajectory = getTrajectory(trajectoryDf)
            chaseTrial(trajectory)

if __name__ == '__main__':
    main()
