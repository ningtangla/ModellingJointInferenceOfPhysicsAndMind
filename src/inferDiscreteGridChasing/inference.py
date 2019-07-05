import pandas as pd
import itertools
import numpy as np

class StayWithinBoundary:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, nextIntendedState):
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY
        return nextX, nextY


def createIndex(chasingAgents, pullingAgents, actionSpace, forceSpace):
    chasingSpace = list(itertools.permutations(chasingAgents))
    pullingSpaceArray = np.unique(list(itertools.permutations(pullingAgents)), axis = 0)
    pullingSpace = [tuple(pullingPair) for pullingPair in pullingSpaceArray.tolist()]
    actionHypo = list(itertools.product(actionSpace, actionSpace, actionSpace))

    forceComboList = [list(itertools.permutations([(0, 0), force, tuple(-np.array(force))])) for force in forceSpace]

    #nested list
    forceCombo = [agentsForce for forceList in forceComboList for agentsForce in forceList]
    forceHypo = list(set(forceCombo))

    iterables = [chasingSpace, pullingSpace, actionHypo, forceHypo]
    index = pd.MultiIndex.from_product(iterables, names=['chasingAgents', 'pullingAgents',
                                                         'action','force'])
    return index  #6* 3* 13 * 64



class GetWolfActionProb:
    def __init__(self, getAgentPosition, heatSeekingActionLikelihood):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingActionLikelihood = heatSeekingActionLikelihood

    def __call__(self, chasingIndices, state, allAgentsAction):
        wolfID = chasingIndices.index('wolf')
        sheepID = chasingIndices.index('sheep')

        getWolfPos = self.getAgentPosition(wolfID)
        getSheepPos = self.getAgentPosition(sheepID)

        getWolfActionLikelihood = self.heatSeekingActionLikelihood(getWolfPos, getSheepPos)
        wolfActionLikelihood = getWolfActionLikelihood(state)

        wolfAction = allAgentsAction[wolfID]
        wolfActionProb = wolfActionLikelihood[wolfAction]

        return wolfActionProb


class GetSheepActionProb:
    def __init__(self, getAgentPosition, heatSeekingActionLikelihood):
        self.getAgentPosition = getAgentPosition
        self.heatSeekingActionLikelihood = heatSeekingActionLikelihood

    def __call__(self, chasingIndices, state, allAgentsAction):
        wolfID = chasingIndices.index('wolf')
        sheepID = chasingIndices.index('sheep')

        getWolfPos = self.getAgentPosition(wolfID)
        getSheepPos = self.getAgentPosition(sheepID)

        getSheepActionLikelihood = self.heatSeekingActionLikelihood(getWolfPos, getSheepPos)
        sheepActionLikelihood = getSheepActionLikelihood(state)

        sheepAction = allAgentsAction[sheepID]
        sheepActionProb = sheepActionLikelihood[sheepAction]

        return sheepActionProb


class GetMasterActionProb:
    def __init__(self, getRandomActionLikelihood):
        self.getRandomActionLikelihood = getRandomActionLikelihood

    def __call__(self, chasingIndices, state, allAgentsAction):
        masterID = chasingIndices.index('master')

        masterActionLikelihood = self.getRandomActionLikelihood(state)
        masterAction = allAgentsAction[masterID]
        masterActionProb = masterActionLikelihood[masterAction]

        return masterActionProb


class GetPulledAgentForceProb:
    def __init__(self, getAgentPosition, getPulledAgentForceLikelihood):
        self.getAgentPosition = getAgentPosition
        self.getPulledAgentForceLikelihood = getPulledAgentForceLikelihood

    def __call__(self, pullingIndices, state, allAgentsForce):

        pulledAgentID, pullingAgentID = np.where(np.array(pullingIndices) == 'pulled')[0]

        getPulledAgentPos = self.getAgentPosition(pulledAgentID)
        getPullingAgentPos = self.getAgentPosition(pullingAgentID)

        pulledAgentState = getPulledAgentPos(state)
        pullingAgentState = getPullingAgentPos(state)

        pullersRelativeLocation = np.array(pullingAgentState) - np.array(pulledAgentState)
        pulledAgentForceLikelihood= self.getPulledAgentForceLikelihood(pullersRelativeLocation)
        
        pulledAgentForce = allAgentsForce[pulledAgentID]
        pulledAgentForceProb = pulledAgentForceLikelihood[pulledAgentForce]
        
        return pulledAgentForceProb
        

def getNoPullAgentForceProb(pullingIndices, state, allAgentsForce):
    noPullAgentID = pullingIndices.index('noPull')
    noPullAgentForce = allAgentsForce[noPullAgentID]

    if not np.all(np.array(noPullAgentForce) == 0):
        noPullAgentForceProb = 0
    else:
        noPullAgentForceProb = 1
        
    return noPullAgentForceProb


class GetTransitionLikelihood:
    def __init__(self, getPulledAgentForceProb, getNoPullAgentForceProb, stayWithinBoundary):
        self.getPulledAgentForceProb = getPulledAgentForceProb
        self.getNoPullAgentForceProb = getNoPullAgentForceProb
        self.stayWithinBoundary = stayWithinBoundary
        
    def __call__(self, pullingIndices, allAgentsForce, allAgentsAction, state, nextState):
        getAgentsForceProb = [self.getPulledAgentForceProb, self.getNoPullAgentForceProb]
        forceLikelihood = np.product([getForceProb(pullingIndices, state, allAgentsForce) for getForceProb in getAgentsForceProb])
        
        calculatedNextState = [np.array(agentState) + np.array(action) + np.array(force) for agentState, action, force in zip(state, allAgentsAction, allAgentsForce)]
        
        expectedNextState = [self.stayWithinBoundary(calculatedAgentNextState) for calculatedAgentNextState in calculatedNextState]

        if np.any(np.array(expectedNextState) != np.array(nextState)):
            transitionLikelihood = 0
        else:
            transitionLikelihood = forceLikelihood
        
        return transitionLikelihood



class InferOneStepDiscreteChasing:
    def __init__(self, getPolicyLikelihood, getTransitionLikelihood):
        self.getPolicyLikelihood = getPolicyLikelihood
        self.getTransitionLikelihood = getTransitionLikelihood

    def __call__(self, state, nextState, inferenceDf):

        hypothesisSpace = inferenceDf.index

        inferenceDf['state'] = [state]* len(hypothesisSpace)

        inferenceDf['nextState'] = [nextState] * len(hypothesisSpace)

        chasingIndicesList = hypothesisSpace.get_level_values('chasingAgents')
        pullingIndicesList = hypothesisSpace.get_level_values('pullingAgents')
        actionList = hypothesisSpace.get_level_values('action')
        forceList = hypothesisSpace.get_level_values('force')

        policyLikelihood = [self.getPolicyLikelihood(chasingIndices, state, allAgentsActions)
                            for chasingIndices, allAgentsActions in zip(chasingIndicesList, actionList)]
        inferenceDf['policyLikelihood'] = policyLikelihood


        transitionLikelihood = [self.getTransitionLikelihood(pullingIndices, allAgentsForce, allAgentsAction, state, nextState)
                                for pullingIndices, allAgentsForce, allAgentsAction in
                                zip(pullingIndicesList, forceList, actionList)]
        inferenceDf['transitionLikelihood'] = transitionLikelihood

        posterior = [singlePrior * singlePolicyLike* singleTransitionLike for singlePrior, singlePolicyLike, singleTransitionLike
                     in zip(inferenceDf['prior'], policyLikelihood, transitionLikelihood)]
        inferenceDf['posterior'] = posterior


        normalizedPosterior = [singlePosterior/ sum(posterior) for singlePosterior in posterior]
        inferenceDf['normalizedPosterior'] = normalizedPosterior

        inferenceDf['marginalizedPosterior'] = inferenceDf.groupby(['chasingAgents', 'pullingAgents'])['normalizedPosterior'].transform('mean')

        return inferenceDf

