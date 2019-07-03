import numpy as np
import random


class CoefficientCotroller():
    def __init__(self, initCoeffs, afterCoeffs, threshold=0):
        self.actionCoeff, self.valueCoeff = initCoeffs
        self.afterActionCoeff, self.afterValueCoeff = afterCoeffs
        self.threshold = threshold
        self.update = False

    def __call__(self, evalDict):
        if evalDict is not None:
            if evalDict["actionLoss"] < self.threshold and not self.update:
                self.actionCoeff = self.afterActionCoeff
                self.valueCoeff = self.afterValueCoeff
                self.update = True
                print("Coefficients of losses Updated to {:.2f} {:.2f}".format(self.actionCoeff, self.valueCoeff))
        return self.actionCoeff, self.valueCoeff


class TrainTerminalController():
    def __init__(self, lossHistorySize, terminalThreshold, validationSize):
        self.lossHistorySize = lossHistorySize
        self.lossHistory = np.ones(self.lossHistorySize)
        self.actionAccuracyHistory = np.zeros(self.lossHistorySize)
        self.valueAccuracyHistory = np.zeros(self.lossHistorySize)
        self.terminalThresHold = terminalThreshold
        self.validationSize = validationSize
        self.validationHistory = np.ones(validationSize)
        self.validationCount = 0

    def __call__(self, evalDict, validationDict, stepNum):
        # loss change terminal
        self.lossHistory[stepNum % self.lossHistorySize] = evalDict["loss"]
        lossChange = np.mean(np.abs(self.lossHistory - np.min(self.lossHistory)))
        self.actionAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["actionAcc"]
        self.valueAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["valueAcc"]
        if lossChange < self.terminalThresHold:
            return True
        # early stop terminal
        # lastValidationLoss = None
        # if stepNum >= self.validationSize:
        #     lastValidationLoss = np.mean(self.validationHistory)
        # self.validationHistory[stepNum % self.validationSize] = validationDict['loss']
        # if lastValidationLoss is not None and lastValidationLoss < np.mean(self.validationHistory):
        #     self.validationCount += 1
        # else:
        #     self.validationCount = 0
        # if self.validationCount >= 10:
            # return True
        return False


class TrainReporter():
    def __init__(self, maxStepNum, reportInterval, tensorBoardSummaryOn=False):
        self.maxStepNum = maxStepNum
        self.reportInterval = reportInterval
        self.tensorBoardSummaryOn = tensorBoardSummaryOn

    def __call__(self, evalDict, stepNum, writer, summary):
        if stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum - 1:
            print("#{} {}".format(stepNum, evalDict))
            if self.tensorBoardSummaryOn:
                writer.add_summary(summary, stepNum)


class LearningRateModifier():
    def __init__(self, initLearningRate, decayRate, decayStep):
        self.initLearningRate = initLearningRate
        self.decayRate = decayRate
        self.decayStep = decayStep

    def __call__(self, globalStep):
        return self.initLearningRate * np.power(self.decayRate, globalStep / self.decayStep)


class SampleBatchFromTrajectory():
    def __init__(self, dataID, preference, stateIndex, distributionIndex, valueIndex, trajNum, stepNum):
        self.dataID = dataID
        self.preference = preference
        self.stateIndex =stateIndex
        self.distIndex = distributionIndex
        self.valueIndex = valueIndex
        self.trajNum = trajNum
        self.stepNum = stepNum

    def __call__(self, trajactories, size=None):
        if self.preference:
            trajectoryLength = [len(traj) for traj in trajactories]
            mean = np.mean(trajectoryLength)
            trajectoryValue = [length/mean for length in trajectoryLength]
            trajValueNorm = np.linalg.norm(trajectoryValue, ord=0)
            trajProb = trajectoryValue / trajValueNorm
        else:
            trajProb = [1/len(trajactories) for _ in range(len(trajactories))]
        choices = np.random.choice(len(trajactories), size=self.trajNum, p=trajProb)
        sampledTrajs = [trajactories[index] for index in choices]
        points = np.concatenate([random.sample(traj, self.stepNum) for traj in sampledTrajs])
        if np.ndim(points) > 2: #TODO: check why the previous sentence generate different dim
            points = points[0]
        flattenPoints = [[np.array(point[self.stateIndex]).flatten(), list(point[self.distIndex][self.dataID].values()), point[self.valueIndex]] for point in points]
        return zip(*flattenPoints)
