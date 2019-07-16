import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import numpy as np
from matplotlib import pyplot as plt

from exec.trajectoriesSaveLoad import LoadTrajectories, GetSavePath, loadFromPickle

class LoadAllTrajectoriesInInterval:
    def __init__(self, loadTrajectoriesForIteration, windowSize):
        self.loadTrajectoriesForIteration = loadTrajectoriesForIteration
        self.windowSize = windowSize

    def __call__(self, startingIterationIndex):
        listOfAllIndexTrajectories = [self.loadTrajectoriesForIteration(iterationIndex) for iterationIndex in
                                     range(startingIterationIndex, startingIterationIndex+self.windowSize)]
        trajectories = [trajectory for iterationIndexTrajectories in listOfAllIndexTrajectories for trajectory in
                        iterationIndexTrajectories]

        return trajectories


class ComputeAverageNumTerminalTrajectories:
    def __init__(self, terminalTimeStep, actionIndex, loadTrajectoriesInInterval):
        self.terminalTimeStep = terminalTimeStep
        self.actionIndex = actionIndex
        self.loadTrajectoriesInInterval = loadTrajectoriesInInterval

    def __call__(self, startingIterationIndex):
        trajectories = self.loadTrajectoriesInInterval(startingIterationIndex)
        print('trajectories', trajectories)
        print('numTrajectories', len(trajectories))
        allTrajectoriesIsTerminal = [trajectory[self.terminalTimeStep][self.actionIndex] is None
                                     for trajectory in trajectories]
        mean = np.mean(allTrajectoriesIsTerminal)

        return mean


def main():
        minIterationIndex = 0
        maxIterationIndex = 999
        windowSize = 100

        # load trajectories
        trajectorySaveParameters = {'bufferSize': 2000, 'learningRate': 0.0001, 'maxRunningSteps': 10,
                                    'miniBatchSize': 64, 'numSimulations': 200, 'numTrajectoriesPerIteration': 1,
                                    'qPosInit': (0, 0, 0, 0), 'qPosInitNoise': 9.7}
        dirName = os.path.dirname(__file__)
        trajectorySaveDirectory = os.path.join(dirName, '..', 'data', 'trainMCTSNNIteratively',
                                               'replayBuffer', 'trajectories')
        trajectoryExtension = '.pickle'
        getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectorySaveParameters)
        loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
        loadTrajectoriesForIteration = lambda iterationIndex: loadTrajectories({'iteration': iterationIndex})
        loadAllTrajectoriesInInterval = LoadAllTrajectoriesInInterval(loadTrajectoriesForIteration, windowSize)

        # function to compute average num trials
        terminalTimeStep = -1
        actionIndex = 1
        computeAverageNumTerminalTrajectories = ComputeAverageNumTerminalTrajectories(terminalTimeStep, actionIndex,
                                                                                      loadAllTrajectoriesInInterval)
        avgNumTerminalTrajectories = [computeAverageNumTerminalTrajectories(iterationIndex) for iterationIndex in
                                      range(minIterationIndex, maxIterationIndex, windowSize)]

        plt.plot(avgNumTerminalTrajectories)
        # plt.xticks(list(range(minIterationIndex, maxIterationIndex, windowSize)))
        xticks = list(range(minIterationIndex, maxIterationIndex, windowSize))
        xticksStr = [str(xtick) for xtick in xticks]
        print(xticksStr)
        plt.xlabel('iteration index')
        plt.xticks(ticks=list(range(len(xticksStr))), labels=xticksStr)
        plt.ylabel('fraction of trajectories where the prey is caught')
        plt.show()


if __name__ == '__main__':
    main()
