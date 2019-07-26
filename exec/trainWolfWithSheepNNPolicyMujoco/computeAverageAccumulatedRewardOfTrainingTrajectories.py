import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState

import numpy as np


def main():
    # load trajectories
    trainDataNumSimulations = 125
    trainDataKillzoneRadius = 2
    trajectoryFixedParameters = {'numSimulations': trainDataNumSimulations, 'killzoneRadius': trainDataKillzoneRadius}
    dirName = os.path.dirname(__file__)
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                           'trainWolfWithSheepNNPolicyMujoco', 'trainingData')
    if not os.path.exists(trajectorySaveDirectory):
        os.makedirs(trajectorySaveDirectory)
    trajectorySaveExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)

    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectories = loadTrajectories({})

    # make measurement
    sheepId = 0
    wolfId = 1
    xPosIndex = [2, 3]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfXPos = GetAgentPosFromState(wolfId, xPosIndex)
    killzoneRadius = 2
    isTerminal = IsTerminal(killzoneRadius, getSheepXPos, getWolfXPos)

    alivePenalty = -0.05
    deathBonus = 1
    rewardFunction = RewardFunctionCompete(alivePenalty, deathBonus, isTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, rewardFunction)
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]

    allMeasurements = [measurementFunction(trajectory) for trajectory in trajectories]
    meanMeasurement = np.mean(allMeasurements)

    print("Mean: ", meanMeasurement)


if __name__ == '__main__':
    main()