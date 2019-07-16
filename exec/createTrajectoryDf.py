import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import numpy as np

from exec.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf, \
    loadFromPickle, saveToPickle
from exec.evaluationFunctions import conditionDfFromParametersDict

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryPath = os.path.join(DIRNAME, '..', 'data',
                                  'evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco',
                                  'trainingData',
                                  'killzoneRadius=0.5_maxRunningSteps=100_numSimulations=100_qPosInitNoise=9.7_qVelInitNoise=5_rolloutHeuristicWeight=0.1_sampleIndex=23.pickle')

    trajectory = loadFromPickle(trajectoryPath)

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

    df = convertTrajectoryToStateDf(trajectory)
    print(df)
    saveToPickle(df, 'df.pickle')

if __name__ == '__main__':
    main()