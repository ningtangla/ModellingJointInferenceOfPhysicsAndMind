import numpy as np
import pickle
import time


class CoefficientCotroller():
    def __init__(self, initCoeffs, afterCoeffs=(1, 1), threshold=0):
        self.actionCoeff, self.valueCoeff = initCoeffs
        self.afterActionCoeff, self.afterValueCoeff = afterCoeffs
        self.threshold = threshold
        self.updated = False

    def __call__(self, evalDict):
        if evalDict is not None:
            if evalDict["actionLoss"] < self.threshold and not self.updated:
                self.actionCoeff = self.afterActionCoeff
                self.valueCoeff = self.afterValueCoeff
                self.updated = True
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


class SavingTrainReporter:
    def __init__(self, reportInterval, getModelSavePath):
        self.reportInterval = reportInterval
        self.getModelSavePath = getModelSavePath
        self.ckptTime = time.time()

    def __call__(self, stepNum, model, reportDict):
        if stepNum == 1 or stepNum % self.reportInterval == 0:
            print(f'{time.time() - self.ckptTime:.1f}s since last ckpt')
            self.ckptTime = time.time()
            print("#{} {}".format(stepNum, reportDict['evaluations']))
            modelSavePath = self.getModelSavePath({'trainSteps': stepNum})
            saver = model.graph.get_collection_ref('saver')[0]
            saver.save(model, modelSavePath, write_meta_graph=False)
            print(f'Model saved in {modelSavePath}')
            reportSavePath = modelSavePath + '.report.pickle'
            with open(reportSavePath, 'wb') as f:
                pickle.dump(reportDict, f)


class LearningRateModifier():
    def __init__(self, initLearningRate, decayRate=None, decayStep=None):
        self.initLearningRate = initLearningRate
        self.decayRate = decayRate
        self.decayStep = decayStep

    def __call__(self, globalStep):
        if self.decayRate is not None and self.decayStep is not None:
            learningRate = self.initLearningRate * np.power(self.decayRate, globalStep / self.decayStep)
        else:
            learningRate = self.initLearningRate
        return learningRate


class PrepareFetches:
    def __init__(self, varNames, layerIndices, findKey):
        self.varNames = varNames
        self.layerIndices = layerIndices
        self.findKey = findKey

    def __call__(self, graph):
        loss_ = graph.get_collection_ref("loss")[0]
        actionLoss_ = graph.get_collection_ref("actionLoss")[0]
        valueLoss_ = graph.get_collection_ref("valueLoss")[0]
        actionAccuracy_ = graph.get_collection_ref("actionAccuracy")[0]
        valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
        evaluations_ = {"loss": loss_, "actionLoss": actionLoss_, "valueLoss": valueLoss_}

        findKeysByVarName = lambda varName: [self.findKey(varName, sectionName, layerNum)
                                             for sectionName, layerNum in self.layerIndices]
        collectionKeys = [findKeysByVarName(varName) for varName in self.varNames]
        getTensorsByKeys = lambda keys: [graph.get_collection(key)[0] for key in keys]
        varNameToTensors = {varName: getTensorsByKeys(keys) for varName, keys in zip(self.varNames, collectionKeys)}
        buildTensorDict = lambda values: dict(zip(self.layerIndices, values))
        varNameToTensorDict = {varName: buildTensorDict(tensors) for varName, tensors in varNameToTensors.items()}

        return {'evaluations': evaluations_, 'variables': varNameToTensorDict}
