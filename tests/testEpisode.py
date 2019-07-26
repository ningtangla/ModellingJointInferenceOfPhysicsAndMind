import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from src.episode import SampleTrajectory, chooseGreedyAction, sampleActionFromActionDist
from src.simple1DEnv import TransitionFunction, Terminal
from collections import Counter
import numpy as np


@ddt
class TestEpisode(unittest.TestCase):
    def setUp(self):
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)
        self.target_state = bound_high
        self.isTerminal = Terminal(self.target_state)
        self.reset = lambda: 0


    # @data(([{0: 1}], [(0, 0, {0: 1})]*5),
    #       ([{1: 0.7, 2: 0.1, 3: 0.2}, {-1: 0.9, 0: 0.1}],
    #        [(0, 1, {1: 0.7, 2: 0.1, 3: 0.2}), (1, -1, {-1: 0.9, 0: 0.1}), (0, 1, {1: 0.7, 2: 0.1, 3: 0.2}), (1, -1, {-1: 0.9, 0: 0.1}), (0, 1, {1: 0.7, 2: 0.1, 3: 0.2})]),
    #       ([{1: 0.6, -1: 0.4}]*5,
    #        [(0, 1, {1: 0.6, -1: 0.4}), (1, 1, {1: 0.6, -1: 0.4}), (2, 1, {1: 0.6, -1: 0.4}), (3, 1, {1: 0.6, -1: 0.4}), (4, 1, {1: 0.6, -1: 0.4})]),
    #       ([{7: 0.7, 10: 0.3}], [(0, 7, {7: 0.7, 10: 0.3}), (7, None, None)]),
    #       ([{2: 0.9, -2: 0.1}, {0: 1}, {3: 0.4, 2: 0.2, 1: 0.2}, {0: 1}, {0: 1}, {2: 0.99, 4: 0.01}],
    #        [(0, 2, {2: 0.9, -2: 0.1}), (2, 3, {3: 0.4, 2: 0.2, 1: 0.2}), (5, 2, {2: 0.99, 4: 0.01}), (7, None, None)]))
    # @unpack
    # def testSampleTrajectory(self, policyArray, groundTruthTraj):
    #     maxRunningSteps = 5
    #     distToAction = agentDistToGreedyAction
    #     sampleTraj = SampleTrajectory(maxRunningSteps, self.transition, self.isTerminal, self.reset, distToAction)
    #     policy = lambda i: policyArray[i]
    #     traj = sampleTraj(policy)
    #     self.assertEqual(traj, groundTruthTraj)


    @data(({(10, 0): 0.5, (-10, 0): 0.5}, 0),
          ({(10, 0): 1/8, (-10, 0): 1/8, (0, 10): 1/8, (0, -10): 1/8, (7, 7): 1/8, (-7, 7): 1/8, (7, -7): 1/8, (-7, -7): 1/8}, 0))
    @unpack
    def testSampleActionFromActionDist(self, actionDist, x):
        numSamples = 1000000
        allSampledActions = [sampleActionFromActionDist(actionDist) for _ in range(numSamples)]
        allActionCounts = Counter(allSampledActions)
        groundTruthProb = list(actionDist.values())
        sampledProb = np.array(list(allActionCounts.values()))/numSamples
        print("SAMPLED PROBABILITIES: ", sampledProb)
        self.assertTrue(np.allclose(groundTruthProb, sampledProb, rtol=0, atol=0.01))


if __name__ == "__main__":
    unittest.main()
