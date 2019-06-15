import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

# Local import
from runPolicyInMujoco import evaluateMeanEpisodeLength, SampleTrajectory


@ddt
class TestRunPolicyIn1DEnv(unittest.TestCase):
    @data(([[(3, 1), (4, 1), (5, 1), (6, -1)]], 4), ([[]], 0), ([[(3, 1), (4, 1), (5, 1), (6, -1)], [(3, 1), (4, 1), (5, 1)], [(3, 1), (4, 1)], [(3, 1)], []], 2))
    @unpack
    def testEvaluateMeanEpisodeLength(self, trajectory, groundTruthMeanEpisodeLength):
        meanEpisodeLength = evaluateMeanEpisodeLength(trajectory)
        self.assertEqual(meanEpisodeLength, groundTruthMeanEpisodeLength)


