import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np

from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf, \
    loadFromPickle, saveToPickle, LoadTrajectories, GetSavePath
from exec.trajectoriesSaveLoad import conditionDfFromParametersDict

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateSupervisedLearning','evaluateTrajectories')

    modelSampleIndex = (0,3)
    trajectoryParameters = {'agentId': 1, 'depth' :888, 'maxRunningSteps': 20, 'learningRate': 0.001, 'miniBatchSize': 256,'numSimulations': 100, 'trainSteps': 40000, 'sampleIndex':modelSampleIndex}


    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryParameters)

    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

    stateIndex = 0
    getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][stateIndex])[0]))
    getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
    getAllLevelValuesRange = {'timeStep': getRangeTrajectoryLength, 'agentId': getRangeNumAgentsFromTrajectory}

    getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 0)
    getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 1)
    getAgentVelXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 2)
    getAgentVelYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 3)
    extractColumnValues = {'xPos': getAgentPosXCoord, 'yPos': getAgentPosYCoord, 'xVel': getAgentVelXCoord,
                           'yVel': getAgentVelYCoord}

    convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(getAllLevelValuesRange, conditionDfFromParametersDict,
                                                            extractColumnValues)

    trajectories = loadTrajectories({})
    numTrajectories = len(trajectories)
    selectedTrajectories = trajectories[0:3]
    selectedDf = [convertTrajectoryToStateDf(trajectory) for trajectory in selectedTrajectories]
    [saveToPickle(df, os.path.join(DIRNAME, '..', '..', 'data',
                                   'evaluateSupervisedLearning', 'evaluateTrajectories',
                                    'agentId=1_depth=888_learningRate=0.001_maxRunningSteps=20_miniBatchSize=256_numSimulations=100_sampleIndex={}_trainSteps=40000_DF.pickle'.format(sampleIndex)))

     for df, sampleIndex in zip(selectedDf, range(numTrajectories))]


if __name__ == '__main__':
    main()