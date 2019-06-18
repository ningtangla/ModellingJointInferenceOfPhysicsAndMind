import os
import pickle
import numpy as np
import pandas as pd

class GetSavePath:
    def __init__(self, dataDirectory, extension):
        self.dataDirectory = dataDirectory
        self.extension = extension

    def __call__(self, parameters):
        nameValueStringPairs = [variable + '=' + str(parameters[variable]) for variable in
                                parameters]

        fileName = '_'.join(nameValueStringPairs) + '.' + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)

        return path


class LoadTrajectories:
    def __init__(self, getSavePath):
        self.getSavePath = getSavePath

    def __call__(self, oneConditionDf):
        savePath = self.getSavePath(oneConditionDf)
        pickleIn = open(savePath, 'rb')
        trajectories = pickle.load(pickleIn)

        return trajectories


class ComputeStatistics:
    def __init__(self, getTrajectories, numTrials, measurementFunction):
        self.getTrajectories = getTrajectories
        self.numTrials = numTrials
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = [self.measurementFunction(trajectory) for trajectory in allTrajectories]
        measurementMean = np.mean(allMeasurements)
        measurementStd = np.std(allMeasurements)

        return pd.Series({'mean': measurementMean, 'std': measurementStd})


