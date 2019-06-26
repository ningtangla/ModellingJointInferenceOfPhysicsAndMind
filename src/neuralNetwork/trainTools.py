import numpy as np


class CoefficientController():
	def __init__(self, initActionCoeff, initValueCoeff, threshold=0):
		self.actionCoeff = initActionCoeff
		self.valueCoeff = initValueCoeff
		self.threshold = threshold
		self.update = False

	def __call__(self, evalDict):
		if evalDict is not None:
			if evalDict["actionLoss"] < self.threshold and not self.update:
				self.actionCoeff = 5
				self.valueCoeff = 1
				self.update = True
				print("Coefficients of losses Updated to {:.2f} {:.2f}".format(self.actionCoeff, self.valueCoeff))
		return self.actionCoeff, self.valueCoeff


class TrainTerminalController():
	def __init__(self, lossHistorySize, terminalThreshold):
		self.lossHistorySize = lossHistorySize
		self.lossHistory = np.ones(self.lossHistorySize)
		self.actionAccuracyHistory = np.zeros(self.lossHistorySize)
		self.valueAccuracyHistory = np.zeros(self.lossHistorySize)
		self.terminalThresHold = terminalThreshold

	def __call__(self, evalDict, stepNum):
		self.lossHistory[stepNum % self.lossHistorySize] = evalDict["loss"]
		lossChange = np.mean(np.abs(self.lossHistory - np.min(self.lossHistory)))
		self.actionAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["actionAcc"]
		self.valueAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["valueAcc"]

		if lossChange < self.terminalThresHold:
			return True
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