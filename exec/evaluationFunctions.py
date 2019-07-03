import os
import numpy as np
import pandas as pd
import glob


class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters, ):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)

        return path


class LoadTrajectories:
    def __init__(self, getSavePath, loadFromPickle):
        self.getSavePath = getSavePath
        self.loadFromPickle = loadFromPickle

    def __call__(self, oneConditionDf):
        indexLevelNames = oneConditionDf.index.names
        parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
        parameters['sampleIndex'] = '*'
        genericSavePath = self.getSavePath(parameters)
        filesNames = glob.glob(genericSavePath)
        trajectories = [self.loadFromPickle(fileName) for fileName in filesNames]

        return trajectories


class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = [self.measurementFunction(trajectory) for trajectory in allTrajectories]
        measurementMean = np.mean(allMeasurements)
        measurementStd = np.std(allMeasurements)

        return pd.Series({'mean': measurementMean, 'std': measurementStd})