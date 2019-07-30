import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import unittest
from ddt import ddt, unpack, data
import numpy as np
from src.episode import sampleAction, chooseGreedyAction, SampleTrajectoryTerminationProbability
from src.simple1DEnv import TransitionFunction, Terminal


@ddt
class TestEpisode(unittest.TestCase):
    def setUp(self):
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)
        self.target_state = bound_high
        self.isTerminal = Terminal(self.target_state)
        self.reset = lambda: 0

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

    @data((0.5, 1), (0.3, 3), (0.7, 2), (1, 1))
    @unpack
    def testSampleTrajectoryTerminationProbability(self, terminationProbability, episodeLength):
        numSamples = 10000
        sampleTrajectoryTerminationProbability = SampleTrajectoryTerminationProbability(terminationProbability,
                                                                                        self.transition, self.isTerminal,
                                                                                        self.reset, chooseGreedyAction)
        policy = lambda state: [{1: 1}]
        trajectories = [sampleTrajectoryTerminationProbability(policy) for _ in range(numSamples)]
        allTrajLengths = [len(trajectory) for trajectory in trajectories]
        fractionTrajectoriesWithGivenLength = len(list(filter(lambda x: x == episodeLength, allTrajLengths)))/numSamples
        fractionGroundTruth = ((1-terminationProbability)**(episodeLength-1))*terminationProbability
        self.assertAlmostEqual(fractionTrajectoriesWithGivenLength, fractionGroundTruth, 1)

if __name__ == "__main__":
    unittest.main()
