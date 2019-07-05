import pandas as pd

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


class InferDiscreteChasingAndDrawDemo:
    def __init__(self, chasingAgents, pullingAgents, actionSpace, forceSpace,
                 createIndex, isInferenceTerminal, inferOneStepDiscreteChasing,
                 drawInferenceResult):
        self.chasingAgents = chasingAgents
        self.pullingAgents = pullingAgents
        self.actionSpace = actionSpace
        self.forceSpace = forceSpace

        self.isInferenceTerminal = isInferenceTerminal
        self.createIndex = createIndex
        self.inferOneStepDiscreteChasing = inferOneStepDiscreteChasing

        self.drawInferenceResult = drawInferenceResult

    def __call__(self, trajectory):
        hypothesisSpace = self.createIndex(self.chasingAgents, self.pullingAgents, self.actionSpace, self.forceSpace)
        prior = [1] * len(hypothesisSpace)
        inferenceDf = pd.DataFrame(prior, index= hypothesisSpace, columns=['prior'])
        inferenceDf['normalizedPosterior'] = [1/ len(hypothesisSpace)] * len(hypothesisSpace)
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

            inferenceDf['prior'] = inferenceDf['marginalizedPosterior']

        inferenceDf = inferenceDf[["prior", "state", "nextState", "policyLikelihood", "transitionLikelihood",
                                   "posterior", "normalizedPosterior", "marginalizedPosterior"]]

        return inferenceDf
