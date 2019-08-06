import numpy as np

class TransitTwoMassPhysics:
    def __init__(self, transitSmallMass, transitLargeMass):
        self.transitSmallMass = transitSmallMass
        self.transitLargeMass = transitLargeMass

    def __call__(self, physics, state, allAgentsActions, nextState):
        if physics == 'smallMass':
            nextIntendedState = self.transitSmallMass(state, allAgentsActions)
        else:
            nextIntendedState = self.transitLargeMass(state, allAgentsActions)

        transitionLikelihood = 1 if np.all(nextIntendedState == nextState) else 0

        return transitionLikelihood


class TransitConstantPhysics:
    def __init__(self, transitAgents):
        self.transitAgents = transitAgents

    def __call__(self, physics, state, allAgentsActions, nextState):
        agentsNextIntendedState = self.transitAgents(state, allAgentsActions)
        transitionLikelihood = 1 if np.allclose(agentsNextIntendedState, nextState) else 0

        # if transitionLikelihood == 1:
        #     print('transition-----------------------------------------------------')
        # else:
        #     print('transition = 0')

        return transitionLikelihood