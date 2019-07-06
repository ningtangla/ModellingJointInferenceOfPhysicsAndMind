import pandas as pd
import numpy as np

class IsInferenceTerminal:
    def __init__(self, thresholdPosterior):
        self.thresholdPosterior = thresholdPosterior

    def __call__(self, inferenceDf):
        rank = inferenceDf.groupby(['chasingAgents', 'pullingAgents']).sum().reset_index()
        largestPosterior = max(rank['normalizedPosterior'])
        if largestPosterior >= self.thresholdPosterior:
            return True
        else:
            return False


class InferOneStepDiscreteChasing:
    def __init__(self, getPolicyLikelihood, getTransitionLikelihood):
        self.getPolicyLikelihood = getPolicyLikelihood
        self.getTransitionLikelihood = getTransitionLikelihood

    def __call__(self, state, nextState, inferenceDf):
        hypothesisSpace = inferenceDf.index

        inferenceDf['state'] = [state] * len(hypothesisSpace)
        inferenceDf['nextState'] = [nextState] * len(hypothesisSpace)

        mindList = hypothesisSpace.get_level_values('chasingAgents')
        physicsList = hypothesisSpace.get_level_values('pullingAgents')
        actionList = hypothesisSpace.get_level_values('action')

        policyLikelihood = [self.getPolicyLikelihood(mind, state, allAgentsActions)
                            for mind, allAgentsActions in zip(mindList, actionList)]
        inferenceDf['policyLikelihood'] = policyLikelihood

        transitionLikelihood = [self.getTransitionLikelihood(physics, state, allAgentsAction, nextState)
                                for physics, allAgentsAction in zip(physicsList, actionList)]
        inferenceDf['transitionLikelihood'] = transitionLikelihood

        posterior = [prior * policy * transition for prior, policy, transition
                     in zip(inferenceDf['prior'], policyLikelihood, transitionLikelihood)]
        inferenceDf['posterior'] = posterior

        inferenceDf['marginalizedPosterior'] = inferenceDf.groupby(['chasingAgents', 'pullingAgents'])['posterior'].transform('sum')

        normalizedPosterior = [singlePosterior / sum(inferenceDf['marginalizedPosterior'])
                               for singlePosterior in inferenceDf['marginalizedPosterior']]
        inferenceDf['normalizedPosterior'] = normalizedPosterior

        return inferenceDf


class InferDiscreteChasingAndDrawDemo:
    def __init__(self, hypothesisSpace, isInferenceTerminal, inferOneStepDiscreteChasing,
                 drawInferenceResult):
        self.hypothesisSpace = hypothesisSpace

        self.isInferenceTerminal = isInferenceTerminal
        self.inferOneStepDiscreteChasing = inferOneStepDiscreteChasing
        self.drawInferenceResult = drawInferenceResult

    def __call__(self, trajectory):
        prior = [1] * len(self.hypothesisSpace)
        inferenceDf = pd.DataFrame(prior, index= self.hypothesisSpace, columns=['prior'])
        inferenceDf['normalizedPosterior'] = [1/ len(self.hypothesisSpace)] * len(self.hypothesisSpace)
        initialState = trajectory[0]
        self.drawInferenceResult(0, initialState, inferenceDf, saveImage=True)

        iterationTime = len(trajectory) - 1
        for index in range(iterationTime):
            print("round", index)
            state = trajectory[index]
            nextState = trajectory[index + 1]
            inferenceDf = self.inferOneStepDiscreteChasing(state, nextState, inferenceDf)
            self.drawInferenceResult(index+1, nextState, inferenceDf, saveImage=True)
            if self.isInferenceTerminal(inferenceDf):
                break
            inferenceDf['prior'] = inferenceDf['normalizedPosterior']

        inferenceDf = inferenceDf[["prior", "state", "nextState", "policyLikelihood", "transitionLikelihood",
                                   "posterior", "marginalizedPosterior", "normalizedPosterior"]]

        rank = inferenceDf.groupby(['chasingAgents', 'pullingAgents']).sum().reset_index()
        largestPosterior = rank.sort_values(['normalizedPosterior'], ascending=False).head(1)[np.array(['chasingAgents', 'pullingAgents', 'normalizedPosterior'])]
        print(largestPosterior)

        return inferenceDf
