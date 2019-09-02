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
    def __init__(self, agentsPowerRatios, transitAgents):
        self.agentsPowerRatios = agentsPowerRatios
        self.transitAgents = transitAgents

    def __call__(self, physics, state, allAgentsActions, nextState):
        weightedAgentsActions = np.array([self.agentsPowerRatios[i] * np.array(allAgentsActions)[i] for i in range(len(self.agentsPowerRatios))])
        agentsNextIntendedState = self.transitAgents(state, weightedAgentsActions)
        transitionLikelihood = 1 if np.allclose(agentsNextIntendedState, nextState) else 0

        # if transitionLikelihood == 1:
        #     print('transition-----------------------------------------------------')
        # else:
        #     print('transition = 0')

        return transitionLikelihood