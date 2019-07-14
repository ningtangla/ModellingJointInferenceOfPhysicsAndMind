import numpy as np

class Policy:
    def __init__(self, wolfPolicy, sheepPolicy, randomPolicy):
        self.wolfPolicy = wolfPolicy
        self.sheepPolicy = sheepPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        if mind[0] == 'random':
            actionLikelihood = [self.randomPolicy(action) for action in allAgentsActions]
            policyLikelihood = np.product(actionLikelihood)

        else:
            wolfID = mind.index('wolf')
            sheepID = mind.index('sheep')

            wolfState = state[wolfID]
            wolfAction = allAgentsActions[wolfID]
            wolfActionLikelihood = self.wolfPolicy(wolfState, wolfAction)

            sheepState = state[sheepID]
            sheepAction = allAgentsActions[sheepID]
            sheepActionLikelihood = self.sheepPolicy(sheepState, sheepAction)
            policyLikelihood = wolfActionLikelihood* sheepActionLikelihood

        return policyLikelihood

