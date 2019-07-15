import pickle
import os
import glob
import pandas as pd


def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)

        return path


def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters


class LoadTrajectories:
    def __init__(self, getSavePath, loadFromPickle):
        self.getSavePath = getSavePath
        self.loadFromPickle = loadFromPickle

    def __call__(self, parameters):
        parameters['sampleIndex'] = '*'
        genericSavePath = self.getSavePath(parameters)
        print("generic save path: ", genericSavePath)
        filesNames = glob.glob(genericSavePath)
        trajectories = [self.loadFromPickle(fileName) for fileName in filesNames]

        return trajectories


class GenerateAllSampleIndexSavePaths:
    def __init__(self, getSavePath):
        self.getSavePath = getSavePath

    def __call__(self, numSamples, pathParameters):
        parametersWithSampleIndex = lambda sampleIndex: dict(list(pathParameters.items()) +
                                                             [('sampleIndex', sampleIndex)])
        allIndexParameters = {sampleIndex: parametersWithSampleIndex(sampleIndex) for sampleIndex in range(numSamples)}
        allSavePaths = {sampleIndex: self.getSavePath(indexParameters) for sampleIndex, indexParameters in
                        allIndexParameters.items()}

        return allSavePaths


class SaveAllTrajectories:
    def __init__(self, saveData, generateAllSampleIndexSavePaths):
        self.saveData = saveData
        self.generateAllSampleIndexSavePaths = generateAllSampleIndexSavePaths

    def __call__(self, trajectories, pathParameters):
        numSamples = len(trajectories)
        allSavePaths = self.generateAllSampleIndexSavePaths(numSamples, pathParameters)
        saveTrajectory = lambda sampleIndex: self.saveData(trajectories[sampleIndex], allSavePaths[sampleIndex])
        [saveTrajectory(sampleIndex) for sampleIndex in range(numSamples)]

        return None


class GetAgentCoordinateFromTrajectoryAndStateDf:
    def __init__(self, stateIndex, coordinates):
        self.stateIndex = stateIndex
        self.coordinates = coordinates

    def __call__(self, trajectory, df):
        timeStep = df.index.get_level_values('timeStep')[0]
        objectId = df.index.get_level_values('agentId')[0]
        coordinates = trajectory[timeStep][self.stateIndex][objectId][self.coordinates]

        return coordinates


class ConvertTrajectoryToStateDf:
    def __init__(self, getAllLevelsValueRange, getDfFromIndexLevelDict, extractColumnValues):
        self.getAllLevelsValueRange = getAllLevelsValueRange
        self.getDfFromIndexLevelDict = getDfFromIndexLevelDict
        self.extractColumnValues = extractColumnValues

    def __call__(self, trajectory):
        indexLevels = {levelName: getLevelValueRange(trajectory) for levelName, getLevelValueRange in
                            self.getAllLevelsValueRange.items()}
        emptyDf = self.getDfFromIndexLevelDict(indexLevels)
        extractTrajectoryInformation = lambda df: pd.Series({columnName: extractColumnValue(trajectory, df) for
                                                             columnName, extractColumnValue in
                                                             self.extractColumnValues.items()})
        stateDf = emptyDf.groupby(list(indexLevels.keys())).apply(extractTrajectoryInformation)

        return stateDf
