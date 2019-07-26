import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

from matplotlib import pyplot as plt
import numpy as np

from exec.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle
from exec.preProcessing import AccumulateRewards
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState


class AverageAccumulatedReward:
    def __init__(self, windowLength, computeAccumulatedReward):
        self.windowLength = windowLength
        self.computeAccumulatedReward = computeAccumulatedReward

    def __call__(self, startIteration):
        allRewards = [self.computeAccumulatedReward(iteration) for iteration in range(startIteration, startIteration+
                                                                                      self.windowLength)]
        meanReward = np.mean(allRewards)

        return meanReward


def main():
    maxIterations = 13300
    windowLength = 250
    plotIterations = list(range(0, maxIterations, windowLength)) + [maxIterations-1]

    # load trajectory for iteration
    dirName = os.path.dirname(__file__)
    maxRunningSteps = 20
    numSimulations = 200
    numTrajectoriesPerIteration = 1
    miniBatchSize = 256
    learningRate = 0.0001
    bufferSize = 2000
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations,
                                 'numTrajectoriesPerIteration': numTrajectoriesPerIteration,
                                 'miniBatchSize': miniBatchSize, 'learningRate': learningRate, 'bufferSize': bufferSize}
    trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data', 'trainMCTSNNIteratively',
                                           'replayBufferStartWithRandomModel10StepsPerIteration', 'trajectories')
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectoryFixedParameters)

    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoryForIteration = lambda iteration: loadTrajectories({'iteration': iteration})[0] if len(loadTrajectories({'iteration': iteration})) > 0 else []

    # function compute accumulated reward of trajectory
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
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0] if len(trajectory) > 0 else 0
    computeAccumulatedReward = lambda iteration: measurementFunction(loadTrajectoryForIteration(iteration))
    averageAccumulatedReward = AverageAccumulatedReward(windowLength, computeAccumulatedReward)

    # make measurement on all trajectories
    y = [averageAccumulatedReward(iteration) for iteration in plotIterations]

    # plot
    plt.plot(y)
    plt.ylabel("Accumulated Rewards")
    plt.xlabel("Iteration")
    plt.title("Accumulated reward of trajectories in Replay buffer\nChasing task with iterative training 10 training steps per iteration")
    plt.xticks(list(range(len(plotIterations))), plotIterations, rotation='vertical')
    plt.show()

if __name__ == '__main__':
    main()