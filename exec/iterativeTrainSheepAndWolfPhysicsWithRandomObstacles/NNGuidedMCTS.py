import numpy as np
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode


class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([state for i in stateBatch for state in i ])
            stateBatch=stateBatch.reshape(1,-1)
            # print(stateBatch)
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist

class ApproximateValue:
    def __init__(self, policyValueNet):
        self.policyValueNet = policyValueNet

    def __call__(self, stateBatch):
        scalarOutput = False
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([state for i in stateBatch for state in i ])
            stateBatch=stateBatch.reshape(1,-1)
            scalarOutput = True
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})
        if scalarOutput:
            valuePrediction = valuePrediction[0][0]
        return valuePrediction
class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction):
        self.chooseAction = chooseAction

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicy]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numSimulations, actionSpace, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numSimulations = numSimulations
        self.actionSpace = actionSpace
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy(selfNNModel)
        transitInMCTS = lambda state, selfAction: self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpace, transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue(selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = MCTS(self.numSimulations, self.selectChild, expand,
                                estimateValue, backup, establishPlainActionDist)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([self.approximatePolicy(NNModel) for NNModel in multiAgentNNModel])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
                                     for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy
        policy = lambda state: [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy
