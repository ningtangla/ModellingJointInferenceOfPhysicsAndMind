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
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'trainNNEscapePolicyMujoco', 'evaluationTrajectories')
    trajectoryParameters = {'evaluationMaxRunningSteps': 25, 'policy': 'mcts', 'trainSteps': 99999}
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
    allDf = [convertTrajectoryToStateDf(trajectory) for trajectory in trajectories]
    selectedDf = allDf[0:5]

    [saveToPickle(df, os.path.join(DIRNAME, '..', '..', 'data',
                                   'trainNNEscapePolicyMujoco', 'videosMCTS', 'sampleIndex{}.pickle'.format(sampleIndex)))
     for df, sampleIndex in
     zip(selectedDf, range(numTrajectories))]


if __name__ == '__main__':
    main()