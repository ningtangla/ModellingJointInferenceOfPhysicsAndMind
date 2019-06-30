import pandas as pd


class IsTerminal:
    def __init__(self, thresholdPosterior, comparisonIndex):
        self.thresholdPosterior = thresholdPosterior
        self.comparisonIndex = comparisonIndex

    def __call__(self, inferenceDf):
        rank = inferenceDf.groupby(self.comparisonIndex).sum().reset_index()
        largestPosterior = max(rank['normalizedPosterior'])
        if largestPosterior >= self.thresholdPosterior:
            return True
        else:
            return False



class InferDiscreteChasing:
    def __init__(self, chasingAgents, pullingAgents, actionSpace, forceSpace,
                 createIndex, isTerminal, inferOneStepDiscreteChasing):
        self.chasingAgents = chasingAgents
        self.pullingAgents = pullingAgents
        self.actionSpace = actionSpace
        self.forceSpace = forceSpace

        self.isTerminal = isTerminal
        self.createIndex = createIndex
        self.inferOneStepDiscreteChasing = inferOneStepDiscreteChasing

    def __call__(self, trajectory):
        hypothesisSpace = self.createIndex(self.chasingAgents, self.pullingAgents, self.actionSpace, self.forceSpace)
        prior = [1] * len(hypothesisSpace)
        inferenceDf = pd.DataFrame(prior, index= hypothesisSpace, columns=['prior'])

        iterationTime = len(trajectory) - 1

        for index in range(iterationTime):
            print("round", index)

            state = trajectory[index]
            nextState = trajectory[index + 1]
            inferenceDf = self.inferOneStepDiscreteChasing(state, nextState, inferenceDf)

            if self.isTerminal(inferenceDf):
                break

            inferenceDf['prior'] = inferenceDf['marginalizedPosterior']


        return inferenceDf


