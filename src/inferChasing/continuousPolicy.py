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
            wolfActionDist = self.wolfPolicy(state)
            wolfActionLikelihood = wolfActionDist.get(wolfAction, 0)

            sheepAction = allAgentsActions[sheepID]
            sheepActionDist = self.sheepPolicy(state)
            sheepActionLikelihood = sheepActionDist.get(sheepAction, 0)

            policyLikelihood = wolfActionLikelihood* sheepActionLikelihood

        return policyLikelihood


class ThreeAgentsPolicyForNN:
    def __init__(self, wolfPolicy, sheepPolicy, randomPolicy, softenPolicy = None, softParam = None):
        self.wolfPolicy = wolfPolicy
        self.sheepPolicy = sheepPolicy
        self.randomPolicy = randomPolicy

        self.softenPolicy = softenPolicy
        self.softParam = softParam

    def __call__(self, mind, state, allAgentsActions):

        if self.softParam is not None:
            wolfPolicy = self.softenPolicy(self.wolfPolicy, self.softParam)
            sheepPolicy = self.softenPolicy(self.sheepPolicy, self.softParam)
            randomPolicy = self.softenPolicy(self.randomPolicy, self.softParam)

        else:
            wolfPolicy = self.wolfPolicy
            sheepPolicy = self.sheepPolicy
            randomPolicy = self.randomPolicy

        wolfID = mind.index('wolf')
        sheepID = mind.index('sheep')

        sheepWolfState = [state[sheepID][:6], state[wolfID][:6]] # outside

        wolfAction = allAgentsActions[wolfID]
        wolfActionDist = wolfPolicy(sheepWolfState)
        wolfActionLikelihood = wolfActionDist.get(wolfAction, 0)

        sheepAction = allAgentsActions[sheepID]
        sheepActionDist = sheepPolicy(sheepWolfState)
        sheepActionLikelihood = sheepActionDist.get(sheepAction, 0)

        randomID = mind.index('random')
        randomAction = allAgentsActions[randomID]
        randomAllActionsLikelihood = randomPolicy(state)
        randomActionLikelihood = randomAllActionsLikelihood.get(randomAction, 0)

        actionLikelihood = [wolfActionLikelihood, sheepActionLikelihood, randomActionLikelihood]
        policyLikelihood = np.product(actionLikelihood)

        return policyLikelihood




class ThreeAgentsPolicyForWolfOnlyMCTS:
    def __init__(self, mctsPolicy, randomPolicy):
        self.mctsPolicy = mctsPolicy
        self.randomPolicy = randomPolicy

    def __call__(self, mind, state, allAgentsActions):
        # print('mind', mind)

        wolfID = mind.index('wolf')
        wolfAction = allAgentsActions[wolfID]
        wolfActionDist = self.mctsPolicy(state)
        wolfActionLikelihood = wolfActionDist.get(wolfAction, 0)

        sheepID = mind.index('sheep')
        sheepAction = allAgentsActions[sheepID]
        sheepActionDist = self.randomPolicy(state)
        sheepActionLikelihood = sheepActionDist.get(sheepAction, 0)

        randomID = mind.index('random')
        randomAction = allAgentsActions[randomID]
        randomAllActionsLikelihood = self.randomPolicy(state)
        randomActionLikelihood = randomAllActionsLikelihood.get(randomAction, 0)

        actionLikelihood = [wolfActionLikelihood, sheepActionLikelihood, randomActionLikelihood]
        policyLikelihood = np.product(actionLikelihood)

        print(policyLikelihood)

        return policyLikelihood




class InferencePolicy:
    def __init__(self, agentsNameList, agentsPolicyList, softenPolicy = None, softParam = None):
        self.agentsNameList = agentsNameList
        self.agentsPolicyList = agentsPolicyList

        self.softenPolicy = softenPolicy
        self.softParam = softParam

    def __call__(self, mind, state, allAgentsActions):

        if self.softParam is not None:
            agentsInferencePolicyList = [self.softenPolicy(policy, self.softParam) for policy in self.agentsPolicyList]
        else:
            agentsInferencePolicyList = self.agentsPolicyList

        actionDistList = [agentPolicy(state) for agentPolicy in agentsInferencePolicyList]
        actionList = [allAgentsActions[mind.index(agentName)] for agentName in self.agentsNameList]

        actionAndActionInfDistPair = zip(actionList, actionDistList)
        getActionLik = lambda action, actionDist: actionDist.get(action, 0)
        likelihoodList = [getActionLik(action, actionDist) for action, actionDist in actionAndActionInfDistPair]

        policyLikelihood = np.product(likelihoodList)

        return policyLikelihood




