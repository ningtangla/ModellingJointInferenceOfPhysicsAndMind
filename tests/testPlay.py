import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
import src.play as play
import src.simple1DEnv as env


@ddt
class TestPlay(unittest.TestCase):
    def setUp(self):
        bound_low = 0
        bound_high = 7
        self.transition = env.TransitionFunction(bound_low, bound_high)
        self.target_state = bound_high
        self.isTerminal = env.Terminal(self.target_state)
        self.reset = lambda: 0

    @data(([0], [(0, 0)]*5),
          ([1, -1], [(0, 1), (1, -1), (0, 1), (1, -1), (0, 1)]),
          ([1, 1, 1, 1, 1], [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
          ([7], [(0, 7), (7, None)]),
          ([2, 0, 3, 0, 0, 2], [(0, 2), (2, 3), (5, 2), (7, None)]))
    @unpack
    def testSampleTrajectory(self, policyArray, groundTruthTraj):
        maxRunningSteps = 5
        sampleTraj = play.SampleTrajectory(maxRunningSteps, self.transition, self.isTerminal, self.reset)
        policy = lambda i: policyArray[i]
        traj = sampleTraj(policy)
        self.assertEqual(traj, groundTruthTraj)

    @data(({0: 1}, 0),
          ({1: 0.2, 0: 0.3, -1: 0.5}, -1),
          ({(0, 0): 0.125, (1, 0): 0.75, (0, 1): 0.125, (1, 1): 0}, (1, 0)),
          ({(1, 0): 0.1, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1, (1, 1): 0.1, (-1, -1): 0.1, (-1, 0): 0.1, (0, -1): 0.3}, (0, -1)))
    @unpack
    def testAgentDistToGreedyAction(self, actionDist, greedyAction):
        selectedAction = play.agentDistToGreedyAction(actionDist)
        self.assertEqual(selectedAction, greedyAction)

    @data(({1: 0.33, 0: 0.33, -1: 0.33, 2: 0.01}, [1, 0, -1], 1000, 200),
          ({(1, 0): 0.1, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1, (1, 1): 0.3, (-1, -1): 0, (-1, 0): 0, (0, -1): 0.3}, [(1, 1), (0, -1)], 1000, 400))
    @unpack
    def testRandomnessInAgentDistToGreedyAction(self, actionDist, greedyActions, rep, minSelected):
        counter = {action: 0 for action in greedyActions}
        for _ in range(rep):
            selectedAction = play.agentDistToGreedyAction(actionDist)
            counter[selectedAction] += 1
        self.assertEqual(sum(counter.values()), rep)
        for action in counter:
            self.assertGreater(counter[action], minSelected)

    @data(([0, 1, 0], [0, 1, 0]),
          ([{0: 0.6, 1: 0.4}], [0]),
          ([{0: 0.6, 1: 0.4}, {0: 0.4, 1: 0.6}, {0: 0, 1: 1}, {0: 0.3, 1: 0.7}], [0, 1, 1, 1]),
          ([{0: 0.6, 1: 0.4}, 1, {0: 0, 1: 1}, 1], [0, 1, 1, 1]),
          ([{(1, 0): 0.1, (0, 1): 0.7, (-1, 0): 0.1, (0, -1): 0.1},
            {(1, 0): 0.1, (0, 1): 0.1, (-1, 0): 0.7, (0, -1): 0.1},
            {(1, 0): 0.7, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1},
            {(1, 0): 0.7, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1}],
           [(0, 1), (-1, 0), (1, 0), (1, 0)]),
          ([{(1, 0): 0.1, (0, 1): 0.7, (-1, 0): 0.1, (0, -1): 0.1},
            (-1, 0),
            (1, 0),
            {(1, 0): 0.7, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1}],
           [(0, 1), (-1, 0), (1, 0), (1, 0)]))
    @unpack
    def testWorldDistToAction(self, dists, actions):
        convertedDists = play.worldDistToAction(play.agentDistToGreedyAction, dists)
        self.assertEqual(convertedDists, actions)

    @data(([{0: 1}], [(0, 0, {0: 1})]*5),
          ([{1: 0.7, 2: 0.1, 3: 0.2}, {-1: 0.9, 0: 0.1}],
           [(0, 1, {1: 0.7, 2: 0.1, 3: 0.2}), (1, -1, {-1: 0.9, 0: 0.1}), (0, 1, {1: 0.7, 2: 0.1, 3: 0.2}), (1, -1, {-1: 0.9, 0: 0.1}), (0, 1, {1: 0.7, 2: 0.1, 3: 0.2})]),
          ([{1: 0.6, -1: 0.4}]*5,
           [(0, 1, {1: 0.6, -1: 0.4}), (1, 1, {1: 0.6, -1: 0.4}), (2, 1, {1: 0.6, -1: 0.4}), (3, 1, {1: 0.6, -1: 0.4}), (4, 1, {1: 0.6, -1: 0.4})]),
          ([{7: 0.7, 10: 0.3}], [(0, 7, {7: 0.7, 10: 0.3}), (7, None, None)]),
          ([{2: 0.9, -2: 0.1}, {0: 1}, {3: 0.4, 2: 0.2, 1: 0.2}, {0: 1}, {0: 1}, {2: 0.99, 4: 0.01}],
           [(0, 2, {2: 0.9, -2: 0.1}), (2, 3, {3: 0.4, 2: 0.2, 1: 0.2}), (5, 2, {2: 0.99, 4: 0.01}), (7, None, None)]))
    @unpack
    def testSampleTrajectoryWithActionDist(self, policyArray, groundTruthTraj):
        maxRunningSteps = 5
        distToAction = play.agentDistToGreedyAction
        sampleTraj = play.SampleTrajectoryWithActionDist(maxRunningSteps, self.transition, self.isTerminal, self.reset, distToAction)
        policy = lambda i: policyArray[i]
        traj = sampleTraj(policy)
        self.assertEqual(traj, groundTruthTraj)


if __name__ == "__main__":
    unittest.main()
