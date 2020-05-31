import numpy as np
import pandas as pd


class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = np.array([self.measurementFunction(trajectory) for trajectory in allTrajectories])
        print(allMeasurements)
        measurementMean = np.mean(allMeasurements, axis = 0)
        measurementStd = np.std(allMeasurements, axis = 0)

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


def conditionDfFromParametersDict(parametersDict):
    levelNames = list(parametersDict.keys())
    levelValues = list(parametersDict.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    conditionDf = pd.DataFrame(index=modelIndex)

    return conditionDf