import numpy as np

class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, agentAction):
        likelihood = {action: 1/len(self.actionSpace) for action in self.actionSpace}
        actionLik = likelihood[agentAction]
        return actionLik


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

            wolfAction = allAgentsActions[wolfID]
            wolfActionLikelihood = self.wolfPolicy(state, wolfAction)

            sheepAction = allAgentsActions[sheepID]
            sheepActionLikelihood = self.sheepPolicy(state, sheepAction)
            policyLikelihood = wolfActionLikelihood* sheepActionLikelihood

        return policyLikelihood

