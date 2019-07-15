import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd

from exec.evaluationFunctions import GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, \
    GenerateAllSampleIndexSavePaths


@ddt
class TestExecEvaluationFunctions(unittest.TestCase):
    def setUp(self):
        self.dataDirectory = 'testData'

    @data(('..', '.txt', {'qPosInit': (1, 2, 3, 4), 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
           '../numSimulations=12_numTrials=23_qPosInit=(1,2,3,4)_trainSteps=2.txt'),
          ('', '.pickle', {'qPosInit': [1, 2, 3, 4], 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
           'numSimulations=12_numTrials=23_qPosInit=[1,2,3,4]_trainSteps=2.pickle'))
    @unpack
    def testGetSavePathWithFixedParameters(self, dataDirectory, extension, fixedParameters, parameters, groundTruthPath):
        getSavePath = GetSavePath(dataDirectory, extension, fixedParameters)
        path = getSavePath(parameters)
        self.assertEqual(path, groundTruthPath)


    @data(('..', '.txt', {'numTrials': 23, 'trainSteps': 2}, '../numTrials=23_trainSteps=2.txt'),
          ('', '.pickle', {'numTrials': 23, 'trainSteps': 2}, 'numTrials=23_trainSteps=2.pickle'))
    @unpack
    def testGetSavePathWithoutFixedParameters(self, dataDirectory, extension, parameters, groundTruthPath):
        getSavePath = GetSavePath(dataDirectory, extension)
        path = getSavePath(parameters)
        self.assertEqual(path, groundTruthPath)


    @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
                                                           'sheepPolicyName': 'MCTS'}))
    @unpack
    def testLoadTrajectoriesNumTrials(self, parameters, fixedParameters):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)

        loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle)
        loadedTrajectories = loadTrajectories(parameters)
        numTrials = len(loadedTrajectories)

        groundTruthNumTrials = parameters['numTrials']
        self.assertEqual(numTrials, groundTruthNumTrials)


    @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
                                                           'sheepPolicyName': 'MCTS'}))
    @unpack
    def testLoadTrajectoriesQPosInit(self, parameters, fixedParameters):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)

        loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle)
        loadedTrajectories = loadTrajectories(parameters)
        initTimeStep = 0
        stateIndex = 0
        qPosIndex = 0
        numQPosEachAgent = 2
        allInitStates = [trajectory[initTimeStep][stateIndex] for trajectory in loadedTrajectories]
        allQPosInit = [initState[:, qPosIndex:qPosIndex+numQPosEachAgent].flatten() for initState in allInitStates]

        groundTruthQPosInit = parameters['qPosInit']

        allTruthValues = np.asarray([np.all(qPosInit == groundTruthQPosInit) for qPosInit in allQPosInit])
        self.assertTrue(np.all(allTruthValues))


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))], names=['numTrials', 'qPosInit'])),
           {'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}))
    @unpack
    def testReadParametersFromDf(self, df, groundTruthParameters):
        parameters = readParametersFromDf(df)
        self.assertEqual(parameters, groundTruthParameters)


    @data((GetSavePath('..', '.pickle', {'numTrials': 25, 'maxRunningSteps': 3}), 3, {'numSimulations': 20},
           {0: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=0.pickle'),
            1: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=1.pickle'),
            2: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=2.pickle')}))
    @unpack
    def testGenerateAllSampleIndexSavePaths(self, getSavePath, numSamples, pathParameters, groundTruthAllPaths):
        generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getSavePath)
        allPaths = generateAllSampleIndexSavePaths(numSamples, pathParameters)
        self.assertEqual(allPaths, groundTruthAllPaths)


if __name__ == "__main__":
    unittest.main()
