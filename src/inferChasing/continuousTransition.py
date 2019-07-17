import numpy as np

class TransitTwoMassPhysics:
    def __init__(self, transitSmallMass, transitLargeMass):
        self.transitSmallMass = transitSmallMass
        self.transitLargeMass = transitLargeMass

    def __call__(self, physics, state, allAgentsActions, nextState):
        sameState = lambda state1, state2: np.all(state1 == state2)
        if physics == 'smallMass':
            nextIntendedState = self.transitSmallMass(state, allAgentsActions)
        else:
            nextIntendedState = self.transitLargeMass(state, allAgentsActions)
            # print(np.all(nextIntendedState == self.transitSmallMass(state, allAgentsActions)))

        transitionLikelihood = 1 if sameState(nextIntendedState, nextState) else 0
        return transitionLikelihood

class TransitConstantPhysics:
    def __init__(self, transitAgents):
        self.transitAgents = transitAgents

    def __call__(self, physics, state, allAgentsActions, nextState):
        agentsNextIntendedState = self.transitAgents(state, allAgentsActions)
        sameState = lambda state1, state2: np.all(state1 == state2)
        transitionLikelihood = 1 if sameState(agentsNextIntendedState, nextState) else 0
        return transitionLikelihood