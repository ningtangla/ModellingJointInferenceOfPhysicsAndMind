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
    forceCombo = [agentsForce for forceList in forceComboList for agentsForce in forceList] # with duplicates
    forceHypo = list(set(forceCombo))

    iterables = [chasingSpace, pullingSpace, actionHypo, forceHypo]
    index = pd.MultiIndex.from_product(iterables, names=['chasingAgents', 'pullingAgents',
                                                         'action','force'])
    return index  #6* 3* 13 * 64


class GetPolicyDistribution:
    def __init__(self, getAgentPosition,
                 getHeatSeekingActionLikelihood,
                 getMasterActionLikelihood):

        self.getAgentPosition = getAgentPosition
        self.getHeatSeekingActionLikelihood = getHeatSeekingActionLikelihood
        self.getMasterActionLikelihood = getMasterActionLikelihood

    def __call__(self, state, chasingIndices):
        wolfID = chasingIndices.index(0)
        sheepID = chasingIndices.index(1)

        getWolfPos = self.getAgentPosition(wolfID)
        getSheepPos = self.getAgentPosition(sheepID)

        getWolfActionLikelihood = self.getHeatSeekingActionLikelihood(getWolfPos, getSheepPos)
        getSheepActionLikelihood = self.getHeatSeekingActionLikelihood(getWolfPos, getSheepPos)

        unorderedLikelihood = [getWolfActionLikelihood, getSheepActionLikelihood, self.getMasterActionLikelihood]
        orderedLikelihood = [unorderedLikelihood[index] for index in chasingIndices]

        getActionLikelihood = lambda state: [computeLikelihood(state) for computeLikelihood in orderedLikelihood]
        actionLikelihoodList = getActionLikelihood(state)

        return actionLikelihoodList


class InferPolicyLikelihood:
    def __init__(self, getPolicyDistribution):
        self.getPolicyDistribution = getPolicyDistribution

    def __call__(self, state, allAgentsAction, chasingIndices):

        actionLikelihoodList = self.getPolicyDistribution(state, chasingIndices)
        targetActionLikelihood = [actionLikelihood[action] for actionLikelihood, action in
                                  zip(actionLikelihoodList, allAgentsAction)]
        policyLikelihood = np.product(targetActionLikelihood)

        return policyLikelihood


class GetPulledAgentForceDistribution:
    def __init__(self, getAgentPosition,getPulledAgentForceLikelihood):
        self.getAgentPosition = getAgentPosition
        self.getPulledAgentForceLikelihood = getPulledAgentForceLikelihood

    def __call__(self, state, pullingIndices):

        pulledAgentID, pullingAgentID = np.where(np.array(pullingIndices) == 0)[0]

        getPulledAgentPos = self.getAgentPosition(pulledAgentID)
        getPullingAgentPos = self.getAgentPosition(pullingAgentID)

        pulledAgentState = getPulledAgentPos(state)
        pullingAgentState = getPullingAgentPos(state)

        pullersRelativeLocation = np.array(pullingAgentState) - np.array(pulledAgentState)

        pulledAgentForceLikelihood= self.getPulledAgentForceLikelihood(pullersRelativeLocation)

        return pulledAgentForceLikelihood


class InferForceLikelihood:
    def __init__(self, getPulledAgentForceDistribution):

        self.getPulledAgentForceDistribution = getPulledAgentForceDistribution

    def __call__(self, state, allAgentsForce, pullingIndices):

        pulledAgentID = pullingIndices.index(0)
        noPullAgentID = pullingIndices.index(1)

        pulledAgentForce = allAgentsForce[pulledAgentID]
        pulledAgentForceLikelihood = self.getPulledAgentForceDistribution(state, pullingIndices)

        targetPulledAgentForceLikelihood = pulledAgentForceLikelihood[pulledAgentForce]

        noPullAgentForce = allAgentsForce[noPullAgentID]

        if not np.all(np.array(noPullAgentForce) == 0):
            noPullAgentForceProb = 0
        else:
            noPullAgentForceProb = 1

        forceLikelihood = targetPulledAgentForceLikelihood * noPullAgentForceProb

        return forceLikelihood


def inferTransitionLikelihood(expectedNextState, observedNextState):
    if np.all(np.array(expectedNextState) == np.array(observedNextState)):
        transitionLikelihood = 1
    else:
        transitionLikelihood = 0

    return transitionLikelihood


class InferOneStepDiscreteChasing:
    def __init__(self, inferPolicyLikelihood, inferForceLikelihood, inferTransitionLikelihood, stayWithinBoundary):
        self.inferPolicyLikelihood = inferPolicyLikelihood
        self.inferForceLikelihood = inferForceLikelihood
        self.inferTransitionLikelihood = inferTransitionLikelihood
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, state, nextState, inferenceDf):

        hypothesisSpace = inferenceDf.index

        inferenceDf['state'] = [state]* len(hypothesisSpace)

        chasingIndicesList = hypothesisSpace.get_level_values('chasingAgents')
        pullingIndicesList = hypothesisSpace.get_level_values('pullingAgents')
        actionList = hypothesisSpace.get_level_values('action')
        forceList = hypothesisSpace.get_level_values('force')

        policyLikelihood = [self.inferPolicyLikelihood(state, allAgentsActions, chasingIndices)
                            for allAgentsActions, chasingIndices in zip(actionList, chasingIndicesList)]
        inferenceDf['policyLikelihood'] = policyLikelihood


        forceLikelihood = [self.inferForceLikelihood(state, allAgentsForces, pullingIndices)
                            for allAgentsForces, pullingIndices in zip(forceList, pullingIndicesList)]
        inferenceDf['forceLikelihood'] = forceLikelihood


        calculateNextState = lambda state, allAgentsActions, allAgentsForces: [np.array(agentState) + np.array(action) + np.array(force) for agentState, action, force in zip(state, allAgentsActions, allAgentsForces)]
        calculatedNextStateList = [calculateNextState(state, allAgentsActions, allAgentsForces)
                             for allAgentsActions, allAgentsForces in zip(actionList, forceList)]
        inferenceDf['calculatedNextState'] = calculatedNextStateList


        expectedNextStateList = [[self.stayWithinBoundary(calculatedAgentNextState) for calculatedAgentNextState in calculatedNextState] for calculatedNextState in calculatedNextStateList]
        inferenceDf['expectedNextState'] = expectedNextStateList


        inferenceDf['nextState'] = [nextState]* len(hypothesisSpace)


        transitionLikelihood = [self.inferTransitionLikelihood(expectedNextState, nextState) for expectedNextState in expectedNextStateList]
        inferenceDf['transitionLikelihood'] = transitionLikelihood


        posterior = [singlePrior * singlePolicyLike* singleForceLike* singleTransitionLike for singlePrior, singlePolicyLike, singleForceLike, singleTransitionLike
                     in zip(inferenceDf['prior'], policyLikelihood, forceLikelihood, transitionLikelihood)]
        inferenceDf['posterior'] = posterior


        normalizedPosterior = [singlePosterior/ sum(posterior) for singlePosterior in posterior]
        inferenceDf['normalizedPosterior'] = normalizedPosterior

        inferenceDf['marginalizedPosterior'] = inferenceDf.groupby(['chasingAgents', 'pullingAgents'])['normalizedPosterior'].transform('mean')

        return inferenceDf

