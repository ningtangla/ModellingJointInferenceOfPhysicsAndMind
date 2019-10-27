import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..'))

import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import pandas as pd

from exec.trajectoriesSaveLoad import LoadTrajectories, GetSavePath, loadFromPickle, readParametersFromDf
from exec.evaluationFunctions import conditionDfFromParametersDict


class PlotHistogram:
    def __init__(self, loadTrajectories):
        self.loadTrajectories = loadTrajectories

    def __call__(self, oneConditionDf, axForDraw):
        trajectories = self.loadTrajectories(oneConditionDf)
        maxRunningSteps = oneConditionDf.index.get_level_values('maxRunningSteps')[0]
        allTrajectoriesLengths = [len(trajectory) for trajectory in trajectories]
        axForDraw.hist(x=allTrajectoriesLengths, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        axForDraw.set_xlabel("Episode length")
        axForDraw.set_ylabel("Frequency")
        axForDraw.set_title('maximum running steps = {}'.format(maxRunningSteps))


def main():
        manipulatedVariables = OrderedDict()
        manipulatedVariables['maxRunningSteps'] = [10, 100]
        toSplitFrame = conditionDfFromParametersDict(manipulatedVariables)

        # load trajectories
        trajectorySaveParameters = {'numSimulations': 100, 'killzoneRadius': 0.5,
                                    'qPosInitNoise': 9.7, 'qVelInitNoise': 5,
                                    'rolloutHeuristicWeight': 0.1}
        dirName = os.path.dirname(__file__)
        trajectorySaveDirectory = os.path.join(dirName, '..', '..', 'data',
                                               'evaluateNNPolicyVsMCTSRolloutAccumulatedRewardWolfChaseSheepMujoco',
                                               'trainingData')
        trajectoryExtension = '.pickle'
        getTrajectorySavePath = GetSavePath(trajectorySaveDirectory, trajectoryExtension, trajectorySaveParameters)
        loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
        loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))

        # function to plot histogram
        plotHistogram = PlotHistogram(loadTrajectoriesFromDf)

        fig = plt.figure()
        numRows = 1
        numColumns = len(manipulatedVariables['maxRunningSteps'])
        plotCounter = 1

        for steps, grp in toSplitFrame.groupby('maxRunningSteps'):
            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            plotHistogram(grp, axForDraw)
            plotCounter += 1

        plt.show()

if __name__ == '__main__':
    main()