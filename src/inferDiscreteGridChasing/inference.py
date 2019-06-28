import pandas as pd
import itertools
import numpy as np


def createIndex(chasingAgents, pullingAgents, actionSpace, forceSpace):
    chasingSpace = list(itertools.permutations(chasingAgents))
    pullingSpace = list(itertools.permutations(pullingAgents))
    actionHypo = list(itertools.product(actionSpace, actionSpace, actionSpace))
    forceHypo = list(itertools.product(forceSpace, forceSpace, forceSpace))

    iterables = [chasingSpace, pullingSpace, actionHypo, forceHypo]
    index = pd.MultiIndex.from_product(iterables, names=['chasingAgents', 'pullingAgents',
                                                         'action','force'])
    return index

class InferPolicyLikelihood:
    def __init__(self, positionIndex, rationalityParam, actionSpace,
                 getWolfPosition, getSheepPosition,
                 getWolfActionLikelihood, getSheepActionLikelihood,
                 getMasterActionLikelihood):

        self.positionIndex = positionIndex
        self.rationalityParam = rationalityParam
        self.actionSpace = actionSpace

        self.getWolfPositionFunc = getWolfPosition
        self.getSheepPositionFunc = getSheepPosition

        self.getWolfActionLikelihoodFunc = getWolfActionLikelihood
        self.getSheepActionLikelihoodFunc = getSheepActionLikelihood

        self.getMasterActionLikelihood = getMasterActionLikelihood

    def __call__(self, state, allAgentsAction, chasingIndices):
        wolfID = chasingIndices.index(0)
        sheepID = chasingIndices.index(1)

        getWolfPos = self.getWolfPositionFunc(wolfID)
        getSheepPos = self.getSheepPositionFunc(sheepID)

        getWolfActionLikelihood = self.getWolfActionLikelihoodFunc(getWolfPos, getSheepPos)
        getSheepActionLikelihood = self.getSheepActionLikelihoodFunc(getWolfPos, getSheepPos)

        unorderedLikelihood = [getWolfActionLikelihood, getSheepActionLikelihood, self.getMasterActionLikelihood]
        orderedLikelihood = [unorderedLikelihood[index] for index in chasingIndices]

        getActionLikelihood = lambda state: [computeLikelihood(state) for computeLikelihood in orderedLikelihood]
        actionLikelihoodList = getActionLikelihood(state)

        targetActionLikelihood = [actionLikelihood[action] for actionLikelihood, action in
                                  zip(actionLikelihoodList, allAgentsAction)]

        policyLikelihood = np.product(targetActionLikelihood)

        return policyLikelihood

class InferForceLikelihood:
    def __init__(self, positionIndex, forceSpace, getPulledAgentPosFunc, getPullingAgentPosFunc,
                 getPulledAgentForceLikelihood):
        self.positionIndex = positionIndex
        self.forceSpace = forceSpace

        self.getPulledAgentPosFunc = getPulledAgentPosFunc
        self.getPullingAgentPosFunc = getPullingAgentPosFunc

        self.getPulledAgentForceLikelihood = getPulledAgentForceLikelihood

    def __call__(self, state, allAgentsForce, pullingIndices):
        pulledAgentID = pullingIndices.index(0)
        noPullAgentID = pullingIndices.index(1)
        pullingAgentID = pullingIndices.index(2)

        pulledAgentForce = allAgentsForce[pulledAgentID]
        pullingAgentForce = allAgentsForce[pullingAgentID]

        if not np.all(np.array(pulledAgentForce) + np.array(pullingAgentForce) == 0):
            return 0

        noPullAgentForce = allAgentsForce[noPullAgentID]
        if not np.all(np.array(noPullAgentForce) == 0):
            noPullAgentForceProb = 0
        else:
            noPullAgentForceProb = 1

        getPulledAgentPos = self.getPulledAgentPosFunc(pulledAgentID)
        getPullingAgentPos = self.getPullingAgentPosFunc(pullingAgentID)

        pulledAgentState = getPulledAgentPos(state)
        pullingAgentState = getPullingAgentPos(state)


        pullersRelativeLocation = np.array(pullingAgentState) - np.array(pulledAgentState)

        pulledAgentForceLikelihood= self.getPulledAgentForceLikelihood(pullersRelativeLocation)

        targetPulledAgentForceLikelihood = pulledAgentForceLikelihood[pulledAgentForce]

        forceLikelihood = targetPulledAgentForceLikelihood * noPullAgentForceProb

        return forceLikelihood


def inferTransitionLikelihood(expectedNextState, observedNextState):

    if np.all(np.array(expectedNextState) == np.array(observedNextState)):
        transitionLikelihood = 1
    else:
        transitionLikelihood = 0

    return transitionLikelihood


class InferOneStepDiscreteChasing:
    def __init__(self, chasingAgents, pullingAgents, actionSpace, forceSpace, createIndex,
                 inferPolicyLikelihood, inferForceLikelihood, inferTransitionLikelihood):
        self.chasingAgents = chasingAgents
        self.pullingAgents = pullingAgents
        self.actionSpace = actionSpace
        self.forceSpace = forceSpace

        self.createIndex = createIndex

        self.inferPolicyLikelihood = inferPolicyLikelihood
        self.inferForceLikelihood = inferForceLikelihood
        self.inferTransitionLikelihood = inferTransitionLikelihood
# pandas.index.get_level_values("levelName")

    def __call__(self, state, nextState):
        hypothesisSpace = createIndex(self.chasingAgents, self.pullingAgents, self.actionSpace, self.forceSpace)
        inferenceDf = pd.DataFrame([[state]]* len(hypothesisSpace), index=hypothesisSpace, columns=['state'])

        chasingIndicesList = [index[0] for index in hypothesisSpace]
        pullingIndicesList = [index[1] for index in hypothesisSpace]
        actionList = [index[2] for index in hypothesisSpace]
        forceList = [index[3] for index in hypothesisSpace]

        policyLikelihood = [self.inferPolicyLikelihood(state, allAgentsActions, chasingIndices)
                            for allAgentsActions, chasingIndices in zip(actionList, chasingIndicesList)]
        inferenceDf['policyLikelihood'] = policyLikelihood

        forceLikelihood = [self.inferForceLikelihood(state, allAgentsForces, pullingIndices)
                            for allAgentsForces, pullingIndices in zip(forceList, pullingIndicesList)]
        inferenceDf['forceLikelihood'] = forceLikelihood

        getExpectedNextState = lambda state, allAgentsActions, allAgentsForces: [np.array(agentState) + np.array(action) + np.array(force)
                     for agentState, action, force in zip(state, allAgentsActions, allAgentsForces)]

        expectedNextStateList = [getExpectedNextState(state, allAgentsActions, allAgentsForces)
                             for allAgentsActions, allAgentsForces in zip(actionList, forceList)]

        inferenceDf['expectedNextState'] = expectedNextStateList

        transitionLikelihood = [self.inferTransitionLikelihood(nextState, state) for nextState in expectedNextStateList]
        inferenceDf['transitionLikelihood'] = transitionLikelihood

        posterior = [singlePolicyLike* singleForceLike* singleTransitionLike for singlePolicyLike, singleForceLike, singleTransitionLike
                     in zip(policyLikelihood, forceLikelihood, transitionLikelihood)]
        inferenceDf['posterior'] = posterior

        return inferenceDf

