import tensorflow as tf
import numpy as np


class Train:
    def __init__(self,
                 maxStepNum, learningRate, lossChangeThreshold, lossHistorySize,
                 reportInterval,
                 summaryOn=False, testData=None):
        self.maxStepNum = maxStepNum
        self.learningRate = learningRate
        self.lossChangeThreshold = lossChangeThreshold
        self.lossHistorySize = lossHistorySize

        self.reportInterval = reportInterval

        self.summaryOn = summaryOn
        self.testData = testData

    def __call__(self, model, trainingData):
        graph = model.graph
        state_, actionLabel_ = graph.get_collection_ref("inputs")
        loss_ = graph.get_collection_ref("loss")[0]
        accuracy_ = graph.get_collection_ref("accuracy")[0]
        trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
        fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
        trainWriter = graph.get_collection_ref('writers')[0]

        stateBatch, actionLabelBatch = trainingData

        lossHistory = np.ones(self.lossHistorySize)
        terminalCond = False

        for stepNum in range(self.maxStepNum):
            if self.summaryOn and (stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum-1 or terminalCond):
                loss, accuracy, _, fullSummary = model.run([loss_, accuracy_, trainOp, fullSummaryOp],
                                                           feed_dict={state_: stateBatch,
                                                                      actionLabel_: actionLabelBatch})
                trainWriter.add_summary(fullSummary, stepNum)
                evaluate(model, self.testData, summaryOn=True, stepNum=stepNum)
            else:
                loss, accuracy, _ = model.run([loss_, accuracy_, trainOp],
                                              feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})

            if stepNum % self.reportInterval == 0:
                print("#{} loss: {}".format(stepNum, loss))

            if terminalCond:
                break

            lossHistory[stepNum % self.lossHistorySize] = loss
            terminalCond = bool(np.std(lossHistory) < self.lossChangeThreshold)

        return model


def evaluate(model, testData, summaryOn=False, stepNum=None):
    graph = model.graph
    state_, actionLabel_ = graph.get_collection_ref("inputs")
    loss_ = graph.get_collection_ref("loss")[0]
    accuracy_ = graph.get_collection_ref("accuracy")[0]
    evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
    testWriter = graph.get_collection_ref('writers')[1]

    stateBatch, actionLabelBatch = testData

    if summaryOn:
        loss, accuracy, evalSummary = model.run([loss_, accuracy_, evalSummaryOp],
                                                feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
        testWriter.add_summary(evalSummary, stepNum)
    else:
        loss, accuracy = model.run([loss_, accuracy_],
                                   feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch})
    return loss, accuracy
