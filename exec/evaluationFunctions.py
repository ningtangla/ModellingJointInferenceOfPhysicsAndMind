import numpy as np
import pandas as pd


class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        [print(len(trajectory)) for trajectory in allTrajectories]
        allMeasurements = [self.measurementFunction(trajectory) for trajectory in allTrajectories]
        measurementMean = np.mean(allMeasurements)
        measurementStd = np.std(allMeasurements)

        return pd.Series({'mean': measurementMean, 'std': measurementStd})

class GenerateInitQPosUniform:
    def __init__(self, minQPos, maxQPos, isTerminal, getResetFromInitQPos):
        self.minQPos = minQPos
        self.maxQPos = maxQPos
        self.isTerminal = isTerminal
        self.getResetFromInitQPos = getResetFromInitQPos

    def __call__(self):
        while True:
            qPosInit = np.random.uniform(self.minQPos, self.maxQPos, 4)
            reset = self.getResetFromInitQPos(qPosInit)
            initState = reset()
            if not self.isTerminal(initState):
                break

        return qPosInit



