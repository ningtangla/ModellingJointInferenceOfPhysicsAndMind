import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

# Local import
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, NullRolloutHeuristic, InitializeChildren
from simple1DEnv import TransitionFunction, RewardFunction, Terminal


@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Env param
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)

        self.action_space = [-1, 1]
        self.num_action_space = len(self.action_space)
        self.actionPrior_func = GetActionPrior(self.action_space)

        step_penalty = -1
        catch_reward = 1
        self.target_state = bound_high
        self.isTerminal = Terminal(self.target_state)

        self.c_init = 0
        self.c_base = 1
        self.calculateScore = CalculateScore(self.c_init, self.c_base)

        self.selectChild = SelectChild(self.calculateScore)
        
        init_state = 3
        level1_0_state = self.transition(init_state, action=0)
        level1_1_state = self.transition(init_state, action=1)
        self.default_actionPrior = 0.5

        self.root = Node(id={1: init_state}, numVisited=1, sumValue=0, actionPrior=self.default_actionPrior, isExpanded=True)
        self.level1_0 = Node(parent=self.root, id={0: level1_0_state}, numVisited=2, sumValue=5, actionPrior=self.default_actionPrior, isExpanded=False)
        self.level1_1 = Node(parent=self.root, id={1: level1_1_state}, numVisited=3, sumValue=10, actionPrior=self.default_actionPrior, isExpanded=False)

        self.initializeChildren = InitializeChildren(self.action_space, self.transition, self.actionPrior_func)
        self.expand = Expand(self.isTerminal, self.initializeChildren)

    @data((0, 1, 0, 1, 0), (1, 1, 0, 1, np.log(3)/2), (1, 1, 1, 1, 1 + np.log(3)/2))
    @unpack
    def testCalculateScore(self, parent_visit_number, self_visit_number, sumValue, actionPrior, groundtruth_score):
        curr_node = Node(numVisited = parent_visit_number)
        child = Node(numVisited = self_visit_number, sumValue = sumValue, actionPrior = actionPrior)
        score = self.calculateScore(curr_node, child)
        self.assertEqual(score, groundtruth_score)

    @data((1, 1, 1, 1, 100))
    @unpack
    def testVisitValueEffectsOnSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue, maxSelectTimes):
        curr_node = Node(numVisited = 0)
        first_child = Node(parent = curr_node, id = 'first', numVisited = firstChildVisited, sumValue = firstChildSumValue, actionPrior = 0.5, isExpanded = False)
        second_child = Node(parent = curr_node, id = 'second', numVisited = secondChildVisited, sumValue = secondChildSumValue, actionPrior = 0.5, isExpanded = False)
        old_child_id = 'none'
        for selectIndex in range(maxSelectTimes):
            new_child = self.selectChild(curr_node)

            if selectIndex % 2 is 1:
                self.assertTrue(new_child.id != old_child_id)

            new_child.sumValue -= 1
            old_child_id = new_child.id

    @data((1, 1, 1, 1, 100))
    @unpack
    def testVisitNumEffectsOnSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue, maxSelectTimes):
        curr_node = Node(numVisited = 1)
        first_child = Node(parent = curr_node, id = 'first', numVisited = firstChildVisited, sumValue = firstChildSumValue, actionPrior = 0.5, isExpanded = False)
        second_child = Node(parent = curr_node, id = 'second', numVisited = secondChildVisited, sumValue = secondChildSumValue, actionPrior = 0.5, isExpanded = False)
        old_child_id = 'none'
        for selectIndex in range(maxSelectTimes):
            new_child = self.selectChild(curr_node)

            if selectIndex % 2 is 1:
                self.assertTrue(new_child.id != old_child_id)

            new_child.numVisited += 1
            old_child_id = new_child.id
     
    @data((3, True, [2, 4]), (0, True, [0, 1]), (7, False, None))
    @unpack
    def testExpand(self, state, has_children, child_states):
        leaf_node = Node(id = {1: state}, numVisited = 1, sumValue = 1, actionPrior = 0.5, isExpanded = False)
        expanded_node = self.expand(leaf_node)

        calc_has_children = (len(expanded_node.children) != 0)
        self.assertEqual(has_children, calc_has_children)

        for child_index, child in enumerate(expanded_node.children):
            cal_child_state = list(child.id.values())[0]
            gt_child_state = child_states[child_index]
            self.assertEqual(gt_child_state, cal_child_state)

    @data((4, 3, 0.125), (3, 4, 0.25))
    @unpack
    def testRollout(self, max_rollout_step, init_state, gt_sumValue):
        max_iteration = 1000

        target_state = 6
        isTerminal = Terminal(target_state)

        catch_reward = 1
        step_penalty = 0
        reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)
        nullRolloutHeuristic = NullRolloutHeuristic()

        rollout_policy = lambda state: np.random.choice(self.action_space)
        leaf_node = Node(id={1: init_state}, numVisited=1, sumValue=0, actionPrior=self.default_actionPrior, isExpanded=True)
        rollout = RollOut(rollout_policy, max_rollout_step, self.transition, reward_func, isTerminal, nullRolloutHeuristic)
        stored_reward = []
        for curr_iter in range(max_iteration):
            stored_reward.append(rollout(leaf_node))
        
        calc_sumValue = np.mean(stored_reward)

        self.assertAlmostEqual(gt_sumValue, calc_sumValue, places=1)

    @data((5, [3,4], [2,1], [8,9], [3,2]))
    @unpack
    def testBackup(self, value, prev_sumValues, prev_visit_nums, new_sumValues, new_visit_nums):
        node_list = []
        for prev_sumValue, prev_visit_num in zip(prev_sumValues, prev_visit_nums):
            node_list.append(Node(id = {1: 4}, numVisited = prev_visit_num, sumValue = prev_sumValue, actionPrior = 0.5, isExpanded = False))

        backup(value, node_list)
        cal_sumValues = [node.sumValue for node in node_list]
        cal_visit_nums = [node.numVisited for node in node_list]

        self.assertTrue(np.all(cal_sumValues == new_sumValues))
        self.assertTrue(np.all(cal_visit_nums == new_visit_nums))


if __name__ == "__main__":
    unittest.main()


