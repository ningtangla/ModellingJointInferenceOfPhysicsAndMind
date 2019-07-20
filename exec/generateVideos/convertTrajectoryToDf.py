import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np

from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf, \
    loadFromPickle, saveToPickle, LoadTrajectories, GetSavePath
from exec.evaluationFunctions import conditionDfFromParametersDict

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'trainMCTSNNIteratively',
                                       'replayBufferStartWithTrainedModel', 'evaluationTrajectories20kTrainSteps')
    trajectoryParameters = {'iteration': 19999, 'maxRunningSteps': 20, 'numTrials': 500, 'policyName': 'NNPolicy',
                            'trainBufferSize': 2000, 'trainLearningRate': 0.0001, 'trainMiniBatchSize': 256,
                            'trainNumSimulations': 200, 'trainNumTrajectoriesPerIteration': 1}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryParameters)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

    stateIndex = 0
    getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][stateIndex])[0]))
    getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
    getAllLevelValuesRange = {'timeStep': getRangeTrajectoryLength, 'agentId': getRangeNumAgentsFromTrajectory}

    getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 2)
    getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 3)
    getAgentVelXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 4)
    getAgentVelYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(stateIndex, 5)
    extractColumnValues = {'xPos': getAgentPosXCoord, 'yPos': getAgentPosYCoord, 'xVel': getAgentVelXCoord,
                           'yVel': getAgentVelYCoord}

    convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(getAllLevelValuesRange, conditionDfFromParametersDict,
                                                            extractColumnValues)

    trajectories = loadTrajectories({})
    numTrajectories = len(trajectories)
    selectedTrajectories = trajectories[0:10]
    selectedDf = [convertTrajectoryToStateDf(trajectory) for trajectory in selectedTrajectories]

    [saveToPickle(df, os.path.join(DIRNAME, '..', '..', 'data',
                                   'trainMCTSNNIteratively', 'replayBufferStartWithTrainedModel',
                                   'evaluationVideos20kTrainSteps', 'sampleIndex{}.pickle'.format(sampleIndex)))
     for df, sampleIndex in
     zip(selectedDf, range(numTrajectories))]


if __name__ == '__main__':
    main()