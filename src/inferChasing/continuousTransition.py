import numpy as np
# transitDeterministically(agentState, agentAction, agentNextState):
class Transition:
    def __init__(self, transitDeterministically):
        self.transitDeterministically = transitDeterministically

    def __call__(self, physics, state, allAgentsActions, nextState):
        stateActionNextStatePair = zip(state, allAgentsActions, nextState)

        agentTransitionLikelihood = [self.transitDeterministically(agentState, agentAction, agentNextState)
                                     for agentState, agentAction, agentNextState in stateActionNextStatePair]
        transitionLikelihood = np.product(agentTransitionLikelihood)
        return transitionLikelihood
