import numpy as np

class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        likelihood = {action: 1/len(self.actionSpace) for action in self.actionSpace}
        return likelihood


class TwoAgentsPolicy:
    def __init__(self, wolfPolicy, sheepPolicy, randomPolicy):
        self.wolfPolicy = wolfPolicy
        self.sheepPolicy = sheepPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        if mind[0] == 'random':
            allActionsLikelihood = self.randomPolicy(state)
            actionLikelihood = [allActionsLikelihood.get(action, 0) for action in allAgentsActions]
            policyLikelihood = np.product(actionLikelihood)

        else:
            wolfID = mind.index('wolf')
            sheepID = mind.index('sheep')

            wolfAction = allAgentsActions[wolfID]
            wolfAllActionsLikelihood = self.wolfPolicy(state)
            wolfActionLikelihood = wolfAllActionsLikelihood.get(wolfAction, 0)

            sheepAction = allAgentsActions[sheepID]
            sheepAllActionsLikelihood = self.sheepPolicy(state)
            sheepActionLikelihood = sheepAllActionsLikelihood.get(sheepAction, 0)

            policyLikelihood = wolfActionLikelihood* sheepActionLikelihood

        return policyLikelihood


class ThreeAgentsPolicy:
    def __init__(self, wolfPolicy, sheepPolicy, randomPolicy):
        self.wolfPolicy = wolfPolicy
        self.sheepPolicy = sheepPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        sheepWolfState = [state[sheepID], state[wolfID]]

        wolfAction = allAgentsActions[wolfID]
        wolfAllActionsLikelihood = self.wolfPolicy(sheepWolfState)
        wolfActionLikelihood = wolfAllActionsLikelihood.get(wolfAction, 0)


        sheepAction = allAgentsActions[sheepID]
        sheepAllActionsLikelihood = self.sheepPolicy(sheepWolfState)
        sheepActionLikelihood = sheepAllActionsLikelihood.get(sheepAction, 0)

        randomID = mind.index('random')
        randomAction = allAgentsActions[randomID]
        randomAllActionsLikelihood = self.randomPolicy(state)
        randomActionLikelihood = randomAllActionsLikelihood.get(randomAction, 0)

        actionLikelihood = [wolfActionLikelihood, sheepActionLikelihood, randomActionLikelihood]
        policyLikelihood = np.product(actionLikelihood)

        return policyLikelihood

