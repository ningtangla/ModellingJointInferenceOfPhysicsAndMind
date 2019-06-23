import sys
import os
sys.path.append("..")
sys.path.append(os.path.join('..', 'exec'))
import unittest
from ddt import ddt, data, unpack
import generateTrainingDataForNN as generateData

@ddt
class TestGenerateData(unittest.TestCase):
    @data(({0: 1}, 0),
          ({1: 0.2, 0: 0.3, -1: 0.5}, -1),
          ({(0, 0): 0.125, (1, 0): 0.75, (0, 1): 0.125, (1, 1): 0}, (1, 0)),
          ({(1, 0): 0.1, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1, (1, 1): 0.1, (-1, -1): 0.1, (-1, 0): 0.1, (0, -1): 0.3}, (0, -1)))
    @unpack
    def testDistToGreedyAction(self, actionDist, greedyAction):
        selectedAction = generateData.distToGreedyAction(actionDist)
        self.assertEqual(selectedAction, greedyAction)

    @data(({1: 0.33, 0: 0.33, -1: 0.33, 2: 0.01}, [1, 0, -1], 1000, 200),
          ({(1, 0): 0.1, (0, 1): 0.1, (-1, 0): 0.1, (0, -1): 0.1, (1, 1): 0.3, (-1, -1): 0, (-1, 0): 0, (0, -1): 0.3}, [(1, 1), (0, -1)], 1000, 400))
    @unpack
    def testRandomnessInGreedyActionFromDist(self, actionDist, greedyActions, rep, minSelected):
        counter = {action: 0 for action in greedyActions}
        for _ in range(rep):
            selectedAction = generateData.distToGreedyAction(actionDist)
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
    def testDistsToActions(self, dists, actions):
        convertedDists = generateData.distsToActions(generateData.distToGreedyAction, dists)
        self.assertEqual(convertedDists, actions)


if __name__ == "__main__":
    unittest.main()
