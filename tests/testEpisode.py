import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import unittest
from ddt import ddt, unpack, data
import numpy as np
from src.episode import sampleAction, chooseGreedyAction


@ddt
class TestSampleAction(unittest.TestCase):
    @data(({(0, 10): 0.01, (8, 8): 0.98, (8, -8): 0.01}, (8, 8)),
          ({(10, 0): 0.9, (8, 8): 0.05, (8, -8): 0.05}, (10, 0)),
          ({(10, 0): 0.1, (8, 8): 0.02, (8, -8): 0.88}, (8, -8)))
    @unpack
    def testSampleAction(self, actionDist, groundTruthAction):
        sampledAction = sampleAction(actionDist)
        self.assertEqual(sampledAction, groundTruthAction)

    @data(({(0, 10): 0.01, (8, 8): 0.98, (8, -8): 0.01}, (8, 8)),
          ({(10, 0): 0.9, (8, 8): 0.05, (8, -8): 0.05}, (10, 0)),
          ({(10, 0): 0.1, (8, 8): 0.02, (8, -8): 0.88}, (8, -8)))
    @unpack
    def testChooseGreedyAction(self, actionDist, groundTruthAction):
        sampledAction = chooseGreedyAction(actionDist)
        self.assertEqual(sampledAction, groundTruthAction)


if __name__ == "__main__":
    unittest.main()
