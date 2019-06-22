import numpy as np
from anytree import AnyNode as Node


class CalculateScore:
    def __init__(self, cInit, cBase):
        self.cInit = cInit
        self.cBase = cBase

    def __call__(self, currentNode, child):
        parentVisitCount = currentNode.numVisited
        selfVisitCount = child.numVisited
        actionPrior = child.actionPrior

        if selfVisitCount == 0:
            uScore = np.inf
            qScore = 0
        else:
            explorationRate = np.log((1 + parentVisitCount + self.cBase) / self.cBase) + self.cInit
            uScore = explorationRate * actionPrior * np.sqrt(parentVisitCount) / float(1 + selfVisitCount)
            qScore = child.sumValue / selfVisitCount

        score = qScore + uScore
        return score


class SelectChild:
    def __init__(self, calculateScore):
        self.calculateScore = calculateScore

    def __call__(self, currentNode):
        scores = [self.calculateScore(currentNode, child) for child in currentNode.children]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selectedChildIndex = np.random.choice(maxIndex)
        selectedChild = currentNode.children[selectedChildIndex]
        return selectedChild


class GetActionPrior:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, currentState):
        actionPrior = {action: 1 / len(self.actionSpace) for action in self.actionSpace}
        return actionPrior


class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextState = self.transition(state, action)
            Node(parent=node, id={action: nextState}, numVisited=0, sumValue=0, actionPrior=initActionPrior[action],
                 isExpanded=False)

        return node


class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if not self.isTerminal(currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode


class HeuristicDistanceToTarget:
    def __init__(self, weight, getTargetPosition, getCurrentPosition):
        self.weight = weight
        self.getTargetPosition = getTargetPosition
        self.getCurrentPosition = getCurrentPosition

    def __call__(self, state):
        terminalPosition = self.getTargetPosition(state)
        currentPosition = self.getCurrentPosition(state)

        distance = np.sqrt(np.sum(np.square(currentPosition - terminalPosition)))
        reward = -self.weight * distance

        return reward


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        reachedTerminal = False
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0
        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            totalRewardForRollout += self.rewardFunction(currentState, action)
            
            if self.isTerminal(currentState):
                reachedTerminal = True
                break

            nextState = self.transitionFunction(currentState, action)
            currentState = nextState

        if not reachedTerminal:
            heuristicReward = self.rolloutHeuristic(currentState)
            totalRewardForRollout += heuristicReward

        return totalRewardForRollout


def backup(value, nodeList):
    for node in nodeList:
        node.sumValue += value
        node.numVisited += 1


def selectNextAction(currentRoot):
    numVisitedForAllChildren = [child.numVisited for child in currentRoot.children]
    maxIndex = np.argwhere(numVisitedForAllChildren == np.max(numVisitedForAllChildren)).flatten()
    selectedChildIndex = np.random.choice(maxIndex)

    action = list(currentRoot.children[selectedChildIndex].id.keys())[0]
    return action


class MCTS:
    def __init__(self, numSimulation, selectChild, expand, rollout, backup, selectNextAction):
        self.numSimulation = numSimulation
        self.selectChild = selectChild
        self.expand = expand
        self.rollout = rollout
        self.backup = backup
        self.selectNextAction = selectNextAction

    def __call__(self, currentState):
        root = Node(id={None: currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                nextNode = self.selectChild(currentNode)
                nodePath.append(nextNode)
                currentNode = nextNode

            leafNode = self.expand(currentNode)
            value = self.rollout(leafNode)
            self.backup(value, nodePath)

        nextAction = self.selectNextAction(root)
        return nextAction


def main():
    pass


if __name__ == "__main__":
    main()