from pygame.color import THECOLORS
import os
import pandas as pd
import sys
import numpy as np
import itertools as it
from collections import OrderedDict
DIRNAME = os.path.dirname(__file__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, \
    loadFromPickle, saveToPickle, LoadTrajectories, GetSavePath, conditionDfFromParametersDict, GetAgentCoordinateFromTrajectoryAndStateDf
from exec.generateExpDemo.trajectory import ScaleTrajectory, AdjustDfFPStoTraj
from exec.generateExpDemo.chasingVisualization import InitializeScreen, DrawBackground, DrawState, ChaseTrialWithTraj, DrawStateWithRope, ChaseTrialWithRopeTraj


def getFileName(parameters, fixedParameters):
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

    # manipulatedVariables['agentId'] = [310]
    #manipulatedVariables['maxRunningSteps'] = [360]
    manipulatedVariables['numSimulations'] = [400]
    manipulatedVariables['killzoneRadius'] = [0.5]
    manipulatedVariables['offset'] = [0, 12]
    manipulatedVariables['beta'] = [0.5]
    manipulatedVariables['masterPowerRatio'] = [0.4]
    manipulatedVariables['numAgents'] = [3]
    manipulatedVariables['pureMCTSAgentId'] = [10]
    # manipulatedVariables['sampleIndex'] = [(0,1)]
    # manipulatedVariables['miniBatchSize'] = [256]#[64, 128, 256, 512]
    # manipulatedVariables['learningRate'] =  [1e-4]#[1e-2, 1e-3, 1e-4, 1e-5]
    # manipulatedVariables['depth'] = [4]#[2,4, 6, 8]
    # manipulatedVariables['trainSteps'] = [20000]#list(range(0,100001, 20000))

    # manipulatedVariables['safeBound'] = [1.5]
    # manipulatedVariables['preyPowerRatio'] =[0.7]
    # manipulatedVariables['wallPunishRatio'] = [0.6]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditionParametersAll = [dict(list(i)) for i in productedValues]

    trajectoryFixedParameters = {}
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'generateExpDemo', 'trajectories')

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    # fuzzySearchParameterNames = ['sampleIndex']
    fuzzySearchParameterNames = ['timeStep', 'maxRunningSteps']
    # fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][stateIndex])[0]))
    getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
    getAllLevelValuesRange = {'timeStep': getRangeTrajectoryLength, 'agentId': getRangeNumAgentsFromTrajectory}

    stateIndex = 0
    getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 0)
    getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 1)
    extractColumnValues = {'xPos': getAgentPosXCoord, 'yPos': getAgentPosYCoord}

    convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(getAllLevelValuesRange, conditionDfFromParametersDict, extractColumnValues)


# convert traj pickle to df
    for conditionParameters in conditionParametersAll:
        trajectories = loadTrajectories(conditionParameters)
        numTrajectories = len(trajectories)
        print(numTrajectories)
        maxNumTrajectories = 50
        numTrajectoryChoose = min(numTrajectories, maxNumTrajectories)
        selectedTrajectories = trajectories[0:numTrajectoryChoose]

        selectedDf = [convertTrajectoryToStateDf(trajectory) for trajectory in selectedTrajectories]

        dataFileName = getFileName(conditionParameters, trajectoryFixedParameters)
        imageSavePath = os.path.join(trajectoryDirectory, dataFileName)
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)

        [saveToPickle(df, os.path.join(imageSavePath, 'sampleIndex={}.pickle'.format(sampleIndex))) for df, sampleIndex in zip(selectedDf, range(numTrajectories))]

# generate demo image
        screenWidth = 800
        screenHeight = 800
        fullScreen = False
        initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
        screen = initializeScreen()
        leaveEdgeSpace = 195
        lineWidth = 4
        xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
        yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]
        screenColor = THECOLORS['black']
        lineColor = THECOLORS['white']

        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

        numOfAgent = int(conditionParameters['numAgents'])
        sheepId = 0
        wolfId = 1
        masterId = 2
        distractorId = 3

        circleSize = 10
        positionIndex = [0, 1]
        numRopePart = 9
        ropePartIndex = list(range(numOfAgent, numOfAgent + numRopePart))

        conditionList = [2]
        conditionValues = [[wolfId, masterId], [wolfId, distractorId], None]

        drawState = DrawState(screen, circleSize, numOfAgent, positionIndex, drawBackground)
        ropeColor = THECOLORS['grey']
        ropeWidth = 4
        drawStateWithRope = DrawStateWithRope(screen, circleSize, numOfAgent, positionIndex, ropePartIndex, ropeColor, ropeWidth, drawBackground)

        colorSpace = [THECOLORS['green'], THECOLORS['red'], THECOLORS['blue'], THECOLORS['yellow']]
        circleColorList = colorSpace[:numOfAgent]

        # for index in range(len(selectedTrajectories)):
        if len(selectedTrajectories) > 0:
            index = 4
            conditionParameters.update({'demoIndex':index})
            saveToPickle([trajectories[index]], getTrajectorySavePath(conditionParameters))
            for condition in conditionList:
                imageFolderName = os.path.join("{}".format(index), 'condition='"{}".format((condition)))
                saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))

                FPS = 60
                chaseTrial = ChaseTrialWithTraj(FPS, circleColorList, drawState, saveImage=True, saveImageDir=saveImageDir)
                chaseTrialWithRope = ChaseTrialWithRopeTraj(FPS, circleColorList, drawStateWithRope, saveImage=True, saveImageDir=saveImageDir)

                rawXRange = [-10, 10]
                rawYRange = [-10, 10]
                scaledXRange = [200, 600]
                scaledYRange = [200, 600]
                scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

                oldFPS = 9
                adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

                getTrajectory = lambda trajectoryDf: scaleTrajectory(adjustFPS(trajectoryDf))
                trajectoryDf = pd.read_pickle(os.path.join(imageSavePath, 'sampleIndex={}.pickle'.format(index)))
                trajectory = getTrajectory(trajectoryDf)
                # chaseTrial(trajectory)
                chaseTrialWithRope(trajectory, conditionValues[condition])


if __name__ == '__main__':
    main()
