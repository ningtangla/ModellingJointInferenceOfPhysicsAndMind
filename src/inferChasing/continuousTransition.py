import numpy as np
# transitDeterministically(agentState, agentAction, agentNextState):
# class Transition:
#     def __init__(self, transitDeterministically):
#         self.transitDeterministically = transitDeterministically
#
#     def __call__(self, physics, state, allAgentsActions, nextState):
#         stateActionNextStatePair = zip(state, allAgentsActions, nextState)
#
#         agentTransitionLikelihood = [self.transitDeterministically(agentState, agentAction, agentNextState)
#                                      for agentState, agentAction, agentNextState in stateActionNextStatePair]
#         transitionLikelihood = np.product(agentTransitionLikelihood)
#         return transitionLikelihood

class Transition:
    def __init__(self, transitAgents):
        self.transitAgents = transitAgents

    def __call__(self, physics, state, allAgentsActions, nextState):
        agentsNextIntendedState = self.transitAgents(state, allAgentsActions)
        sameState = lambda state1, state2: np.all(np.round(state1, 4) == np.round(state2, 4))
        transitionLikelihood = 1 if sameState(agentsNextIntendedState, nextState) else 0
        return transitionLikelihood