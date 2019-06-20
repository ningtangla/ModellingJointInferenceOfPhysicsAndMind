import sys
import os
sys.path.append(os.path.join('..', 'exec'))
sys.path.append(os.path.join('..', 'src', 'algorithms'))
sys.path.append(os.path.join('..', 'exec', 'testMCTSUniformVsNNPriorChaseMujoco'))
sys.path.append(os.path.join('..', 'src'))
sys.path.append(os.path.join('..', 'src', 'sheepWolf'))
sys.path.append(os.path.join('..', 'src', 'neuralNetwork'))

import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd

from testMCTSUniformVsNNPriorChaseMujoco import GetNonUniformPriorAtSpecificState
from mcts import GetActionPrior
from trainNeuralNet import ActionToOneHot, PreProcessTrajectories
from evaluationFunctions import GetSavePath, LoadTrajectories


class GetNonUniformPrior:
    def __init__(self, actionSpace, preferredAction, priorForPreferredAction):
        self.actionSpace = actionSpace
        self.preferredAction = preferredAction
        self.priorForPreferredAction = priorForPreferredAction

    def __call__(self, state):
        actionPrior = {action: (1-self.priorForPreferredAction) / (len(self.actionSpace)-1) for action in self.actionSpace}
        actionPrior[self.preferredAction] = self.priorForPreferredAction

        return actionPrior


@ddt
class TestNNMCTS(unittest.TestCase):
    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.9),
          (np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (-10, 0), 0.1 / 7),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (10, 0), 0.125),
          (np.asarray([[4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), (7, 7), 0.125))
    @unpack
    def testGetNonUniformPriorAtSpecificState(self, state, action, groundTruthActionPriorForAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        preferredAction = (10, 0)
        priorForPreferredAction = 0.9
        getNonUniformPrior = GetNonUniformPrior(actionSpace, preferredAction, priorForPreferredAction)
        getUniformPrior = GetActionPrior(actionSpace)
        specificState = [[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]
        getNonUniformPriorAtSpecificState = GetNonUniformPriorAtSpecificState(getNonUniformPrior, getUniformPrior, specificState)

        actionPrior = getNonUniformPriorAtSpecificState(state)
        self.assertAlmostEqual(actionPrior[action], groundTruthActionPriorForAction)


    @data(((10, 0), [1, 0, 0, 0, 0, 0, 0, 0]), ((7, 7), [0, 1, 0, 0, 0, 0, 0, 0]), ((1, 2), [0, 0, 0, 0, 0, 0, 0, 0]),
          ((-7, -7), [0, 0, 0, 0, 0, 1, 0, 0]))
    @unpack
    def testActionToOneHot(self, action, groundTruthOneHotAction):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        oneHotAction = actionToOneHot(action)

        self.assertEqual(oneHotAction, groundTruthOneHotAction)


    @data(([[([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)]), ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], None)]],
          [([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])]),
          ([[([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)]),
             ([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]], [(10, 0), (0, 0)])]],
           [([-4, 0, -4, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
            ([-3, 0, -3, 0, 0, 0, 4, 0, 4, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])]))
    @unpack
    def testPreProcessTrajectories(self, trajectories, groundTruthStateActionPairsProcessed):
        sheepId = 0
        actionIndex = 1
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        actionToOneHot = ActionToOneHot(actionSpace)
        preProcessTrajectories = PreProcessTrajectories(sheepId, actionIndex, actionToOneHot)
        stateActionPairsProcessed = preProcessTrajectories(trajectories)

        self.assertEqual(stateActionPairsProcessed, groundTruthStateActionPairsProcessed)


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


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))], names=['numTrials', 'qPosInit'])), {'maxRunningSteps': 15, 'numSimulations': 200, 'sheepPolicyName': 'mcts'}))
    @unpack
    def testLoadTrajectoriesNumTrials(self, oneConditionDf, fixedParameters):
        getSavePath = GetSavePath('../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories', '.pickle', fixedParameters)

        loadTrajectories = LoadTrajectories(getSavePath)
        loadedTrajectories = loadTrajectories(oneConditionDf)
        numTrials = len(loadedTrajectories)

        groundTruthNumTrials = oneConditionDf.index.get_level_values('numTrials')[0]

        self.assertEqual(numTrials, groundTruthNumTrials)


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))], names=['numTrials', 'qPosInit'])), {'maxRunningSteps': 15, 'numSimulations': 200, 'sheepPolicyName': 'mcts'}))
    @unpack
    def testLoadTrajectoriesQPosInit(self, oneConditionDf, fixedParameters):
        getSavePath = GetSavePath('../data/testMCTSUniformVsNNPriorChaseMujoco/trajectories', '.pickle', fixedParameters)

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




