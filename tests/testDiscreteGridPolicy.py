import sys
sys.path.append('../src/sheepWolf')

import unittest
from ddt import ddt, data, unpack

from discreteGridPolicyFunctions import *
from calculateAngleFunction import *
from discreteGridWrapperFunctions import LocateAgent


@ddt
class TestPolicyFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(-1, 0), (1,0), (0, 1), (0, -1),(0, 0)]
        self.rationalityParam = 0.9
        self.lowerBoundAngle = 0
        self.upperBoundAngle = np.pi/2

        self.actHeatSeeking = ActHeatSeeking(self.actionSpace, calculateAngle, self.lowerBoundAngle, self.upperBoundAngle)

        self.wolfID = 0
        self.sheepID = 1
        self.masterID = 2
        self.positionIndex = [0, 1]

        self.locateWolf = LocateAgent(self.wolfID, self.positionIndex)
        self.locateSheep = LocateAgent(self.sheepID, self.positionIndex)
        self.locateMaster = LocateAgent(self.masterID, self.positionIndex)

    @data(((3,2),[(1,0), (0,1)],[(-1, 0), (0, -1), (0,0)]),
           ((0,-1), [(0, -1)],[(-1, 0), (1, 0), (0, 1), (0, 0)]))
    @unpack 
    def testHeatSeekingProperAction(self, heatSeekingDirection, trueChosenActions, trueUnchosenActions):
        actionLists = self.actHeatSeeking(heatSeekingDirection)
        chosenActions = actionLists[0]
        unchosenActions = actionLists[1]
        self.assertEqual(chosenActions, trueChosenActions)
        self.assertEqual(unchosenActions, trueUnchosenActions)

    @data(
        ([(2, 3),(4, 2)], {(-1, 0): 0.1/3, (1,0): 0.45, (0, 1): 0.1/3, (0, -1): 0.45, (0,0): 0.1/3}),
        ([(2, 2), (4, 2)], {(-1, 0): 0.1 / 4, (1, 0): 0.9, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}),
        ([(4, 2),(5, 1)], {(-1, 0): 0.1 / 3, (1, 0): 0.45, (0, 1): 0.1 / 3, (0, -1): 0.45, (0, 0): 0.1 / 3}),
        ([(5, 2), (5, 1)], {(-1, 0): 0.1 / 4, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.9, (0, 0): 0.1 / 4}),
        ([(4, 2), (2, 3)], {(-1, 0): 0.45, (1, 0): 0.1 / 3, (0, 1): 0.45, (0, -1): 0.1 / 3, (0, 0): 0.1 / 3}),
        ([(4, 2), (2, 2)], {(-1, 0): 0.9, (1, 0): 0.1 / 4, (0, 1): 0.1 / 4, (0, -1): 0.1 / 4, (0, 0): 0.1 / 4}))
    @unpack
    def testHeatSeekingPolicy(self, state, trueActionLikelihood):

        getHeatSeekingPolicy = HeatSeekingPolicy(self.rationalityParam, self.actHeatSeeking, self.locateWolf, self.locateSheep)

        iterationTime = 10000
        trueActionLikelihoodPair = zip(trueActionLikelihood.keys(), trueActionLikelihood.values())
        trueActionCount = {action: trueActionProb * iterationTime for
                                  action, trueActionProb in trueActionLikelihoodPair}
        intendedActionList = [getHeatSeekingPolicy(state) for _ in range(iterationTime)]

        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action],intendedActionList.count(action), delta=200)



    def testRandomActionPolicy(self):
        state = [[1,2], [2,3], [3,4]]
        getRandomPolicy = RandomActionPolicy(self.actionSpace)

        iterationTime = 10000
        trueActionCount = {action: 1/5 * iterationTime for action in self.actionSpace}
        intendedActionList = [getRandomPolicy(state) for _ in range(iterationTime)] 
        
        for action in trueActionCount.keys():
            self.assertAlmostEqual(trueActionCount[action],intendedActionList.count(action), delta=200)



    def tearDown(self):
        pass


if __name__ == "__main__":
    policyTest = unittest.TestLoader().loadTestsFromTestCase(TestPolicyFunctions)
    unittest.TextTestRunner(verbosity=2).run(policyTest)


