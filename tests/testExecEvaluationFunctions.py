import sys
import os
sys.path.append('..')

import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd

from exec.evaluationFunctions import GetSavePath, LoadTrajectories


@ddt
class TestExecEvaluationFunctions(unittest.TestCase):
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


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))],
                                                        names=['numTrials', 'qPosInit'])), {'maxRunningSteps': 15,
                                                                                            'numSimulations': 200,
                                                                                            'sheepPolicyName': 'MCTS'}))
    @unpack
    def testLoadTrajectoriesNumTrials(self, oneConditionDf, fixedParameters):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)

        loadTrajectories = LoadTrajectories(getSavePath)
        loadedTrajectories = loadTrajectories(oneConditionDf)
        numTrials = len(loadedTrajectories)

        groundTruthNumTrials = oneConditionDf.index.get_level_values('numTrials')[0]

        self.assertEqual(numTrials, groundTruthNumTrials)


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))],
                        names=['numTrials', 'qPosInit'])), {'maxRunningSteps': 15, 'numSimulations': 200,
                                                            'sheepPolicyName': 'MCTS'}))
    @unpack
    def testLoadTrajectoriesQPosInit(self, oneConditionDf, fixedParameters):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)

        loadTrajectories = LoadTrajectories(getSavePath)
        loadedTrajectories = loadTrajectories(oneConditionDf)
        initTimeStep = 0
        stateIndex = 0
        qPosIndex = 0
        numQPosEachAgent = 2
        allInitStates = [trajectory[initTimeStep][stateIndex] for trajectory in loadedTrajectories]
        allQPosInit = [initState[:, qPosIndex:qPosIndex+numQPosEachAgent].flatten() for initState in allInitStates]

        groundTruthQPosInit = np.asarray(oneConditionDf.index.get_level_values('qPosInit')[0])

        allTruthValues = np.asarray([np.all(qPosInit == groundTruthQPosInit) for qPosInit in allQPosInit])

        self.assertTrue(np.all(allTruthValues))

if __name__ == "__main__":
    unittest.main()
