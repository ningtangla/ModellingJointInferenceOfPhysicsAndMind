import numpy as np


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


class ThreeAgentsPolicyForNN:
    def __init__(self, wolfPolicy, sheepPolicy, randomPolicy):
        self.wolfPolicy = wolfPolicy
        self.sheepPolicy = sheepPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        print('mind', mind)

        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        sheepWolfState = [state[sheepID][:6], state[wolfID][:6]]

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


class ThreeAgentsPolicyForMCTS:
    def __init__(self, mctsPolicy, randomPolicy):
        self.mctsPolicy = mctsPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        print('mind', mind)

        wolfID = mind.index('wolf')
        wolfAction = allAgentsActions[wolfID]
        wolfAllActionsLikelihood = self.mctsPolicy(state)
        wolfActionLikelihood = wolfAllActionsLikelihood.get(wolfAction, 0)

        sheepID = mind.index('sheep')
        sheepAction = allAgentsActions[sheepID]
        sheepAllActionsLikelihood = self.randomPolicy(state)
        sheepActionLikelihood = sheepAllActionsLikelihood.get(sheepAction, 0)

        randomID = mind.index('random')
        randomAction = allAgentsActions[randomID]
        randomAllActionsLikelihood = self.randomPolicy(state)
        randomActionLikelihood = randomAllActionsLikelihood.get(randomAction, 0)

        actionLikelihood = [wolfActionLikelihood, sheepActionLikelihood, randomActionLikelihood]
        policyLikelihood = np.product(actionLikelihood)

        print(policyLikelihood)

        return policyLikelihood

