import tensorflow as tf
import numpy as np
import random


class GenerateModelSeparateLastLayer:
	def __init__(self, numStateSpace, numActionSpace, learningRate, regularizationFactor,
				 valueRelativeErrBound=0.01, seed=128):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.learningRate = learningRate
		self.regularizationFactor = regularizationFactor
		self.valueRelativeErrBound = valueRelativeErrBound
		self.seed = seed

	def __call__(self, hiddenWidths, summaryPath="./tbdata"):
		print("Generating Policy Net with hidden layers: {}".format(hiddenWidths))
		graph = tf.Graph()
		with graph.as_default():
			if self.seed is not None:
				tf.set_random_seed(self.seed)

			with tf.name_scope("inputs"):
				state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
				actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
				valueLabel_ = tf.placeholder(tf.float32, [None, 1], name="valueLabel_")
				actionLossCoef_ = tf.constant(50, dtype=tf.float32)
				valueLossCoef_ = tf.constant(1, dtype=tf.float32)
				tf.add_to_collection("inputs", state_)
				tf.add_to_collection("inputs", actionLabel_)
				tf.add_to_collection("inputs", valueLabel_)
				tf.add_to_collection("lossCoefs", actionLossCoef_)
				tf.add_to_collection("lossCoefs", valueLossCoef_)

			with tf.name_scope("hidden"):
				initWeight = tf.random_uniform_initializer(-0.03, 0.03)
				initBias = tf.constant_initializer(0.001)

				fc1 = tf.layers.Dense(units=hiddenWidths[0], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				a1_ = fc1(state_)
				w1_, b1_ = fc1.weights
				tf.summary.histogram("w1", w1_)
				tf.summary.histogram("b1", b1_)
				tf.summary.histogram("a1", a1_)

				a_ = a1_
				for i in range(1, len(hiddenWidths)-1):
					fc = tf.layers.Dense(units=hiddenWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
					aNext_ = fc(a_)
					a_ = aNext_
					w_, b_ = fc.weights
					tf.summary.histogram("w{}".format(i+1), w_)
					tf.summary.histogram("b{}".format(i+1), b_)
					tf.summary.histogram("a{}".format(i+1), a_)

				fcAction = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aAction_ = fcAction(a_)
				fcLastAction = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				allActionActivation_ = fcLastAction(aAction_)
				wLastAction_, bLastAction_ = fcLastAction.weights
				tf.summary.histogram("wLastAction", wLastAction_)
				tf.summary.histogram("bLastAction", bLastAction_)
				tf.summary.histogram("allActionActivation", allActionActivation_)

				fcValue = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aValue_ = fcValue(a_)
				fcLastValue = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				valueActivation_ = fcLastValue(aValue_)
				wLastValue_, bLastValue_ = fcLastValue.weights
				tf.summary.histogram("wLastValue", wLastValue_)
				tf.summary.histogram("bLastValue", bLastValue_)
				tf.summary.histogram("value", valueActivation_)

			with tf.name_scope("outputs"):
				actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
				actionLoss_ = tf.reduce_mean(cross_entropy, name='actionLoss_')
				tf.add_to_collection("actionDist", actionDistribution_)
				tf.add_to_collection("actionLoss", actionLoss_)
				actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

				actionIndices_ = tf.argmax(actionDistribution_, axis=1)
				actionLabelIndices_ = tf.argmax(actionLabel_, axis=1)
				actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, actionLabelIndices_), tf.float32))
				tf.add_to_collection("actionIndices", actionIndices_)
				tf.add_to_collection("actionAccuracy", actionAccuracy_)
				actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

				valuePrediction_ = valueActivation_
				valueLoss_ = tf.sqrt(tf.losses.mean_squared_error(valueLabel_, valuePrediction_))
				tf.add_to_collection("valuePrediction", valuePrediction_)
				tf.add_to_collection("valueLoss", valueLoss_)
				valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

				relativeErrorBound_ = tf.constant(self.valueRelativeErrBound)
				relativeValueError_ = tf.cast((tf.abs((valuePrediction_ - valueLabel_) / valueLabel_)), relativeErrorBound_.dtype)
				valueAccuracy_ = tf.reduce_mean(tf.cast(tf.less(relativeValueError_, relativeErrorBound_), tf.float64))
				tf.add_to_collection("valueAccuracy", valueAccuracy_)
				valueAccuracySummary = tf.summary.scalar("valueAccuracy", valueAccuracy_)

			with tf.name_scope("train"):
				l2RegularizationLoss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.regularizationFactor
				loss_ = actionLossCoef_*actionLoss_ + valueLossCoef_*valueLoss_ + l2RegularizationLoss_
				tf.add_to_collection("loss", loss_)
				tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)
				lossSummary = tf.summary.scalar("loss", loss_)

				optimizer = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_')
				gradVarPairs_ = optimizer.compute_gradients(loss_)
				trainOp = optimizer.apply_gradients(gradVarPairs_)
				tf.add_to_collection(tf.GraphKeys.TRAIN_OP, trainOp)

				gradients_ = [tf.reshape(grad, [1, -1]) for (grad, _) in gradVarPairs_]
				gradTensor_ = tf.concat(gradients_, 1)
				gradNorm_ = tf.norm(gradTensor_)
				tf.add_to_collection("gradNorm", gradNorm_)
				tf.summary.histogram("gradients", gradTensor_)
				tf.summary.scalar("gradNorm", gradNorm_)

			fullSummary = tf.summary.merge_all()
			evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary,
											actionAccuracySummary, valueAccuracySummary])
			tf.add_to_collection("summaryOps", fullSummary)
			tf.add_to_collection("summaryOps", evalSummary)

			trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
			testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
			tf.add_to_collection("writers", trainWriter)
			tf.add_to_collection("writers", testWriter)

			saver = tf.train.Saver()
			tf.add_to_collection("saver", saver)

			model = tf.Session(graph=graph)
			model.run(tf.global_variables_initializer())

		return model


class Train:
	def __init__(self, maxStepNum, batchSize, terimnalController, CoefficientController, trainReporter):
		self.maxStepNum = maxStepNum
		self.batchSize = batchSize
		self.reporter = trainReporter
		self.terminalController = terimnalController
		self.CoefficientController = CoefficientController

	def __call__(self, model, trainingData):
		graph = model.graph
		state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
		actionLossCoef_, valueLossCoef_ = graph.get_collection_ref("lossCoefs")
		loss_ = graph.get_collection_ref("loss")[0]
		actionLoss_ = graph.get_collection_ref("actionLoss")[0]
		valueLoss_ = graph.get_collection_ref("valueLoss")[0]
		actionAccuracy_ = graph.get_collection_ref("actionAccuracy")[0]
		valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
		trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
		fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
		trainWriter = graph.get_collection_ref('writers')[0]
		fetches = [{"loss": loss_, "actionLoss": actionLoss_, "actionAcc": actionAccuracy_, "valueLoss": valueLoss_, "valueAcc": valueAccuracy_},
				   trainOp, fullSummaryOp]

		evalDict = None
		trainingDataList = list(zip(*trainingData))

		for stepNum in range(self.maxStepNum):
			if self.batchSize is None:
				stateBatch, actionLabelBatch, valueLabelBatch = trainingData
			else:
				stateBatch, actionLabelBatch, valueLabelBatch = sampleData(trainingDataList, self.batchSize)
			actionLossCoef, valueLossCoef = self.CoefficientController(evalDict)
			evalDict, _, summary = model.run(fetches, feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch, valueLabel_: valueLabelBatch,
																 actionLossCoef_: actionLossCoef, valueLossCoef_: valueLossCoef})

			self.reporter(evalDict, stepNum, trainWriter, summary)

			if self.terminalController(evalDict, stepNum):
				break

		return model


def evaluate(model, testData, summaryOn=False, stepNum=None):
	graph = model.graph
	state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
	loss_ = graph.get_collection_ref("loss")[0]
	actionLoss_ = graph.get_collection_ref("actionLoss")[0]
	valueLoss_ = graph.get_collection_ref("valueLoss")[0]
	actionAccuracy_ = graph.get_collection_ref("actionAccuracy")[0]
	valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
	evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
	testWriter = graph.get_collection_ref('writers')[1]
	fetches = [{"actionLoss": actionLoss_, "actionAcc": actionAccuracy_, "valueLoss": valueLoss_, "valueAcc": valueAccuracy_},
			   evalSummaryOp]

	stateBatch, actionLabelBatch, valueLabelBatch = testData
	evalDict, summary = model.run(fetches, feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch, valueLabel_: valueLabelBatch})
	if summaryOn:
		testWriter.add_summary(summary, stepNum)
	return evalDict

def sampleData(data, batchSize):
	batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
	return batch


def saveVariables(model, path):
	graph = model.graph
	saver = graph.get_collection_ref("saver")[0]
	saver.save(model, path)
	print("Model saved in {}".format(path))


def restoreVariables(model, path):
	graph = model.graph
	saver = graph.get_collection_ref("saver")[0]
	saver.restore(model, path)
	print("Model restored from {}".format(path))
	return model


class ApproximatePolicy():
	def __init__(self, model, actionSpace):
		self.actionSpace = actionSpace
		self.model = model

	def __call__(self, stateBatch):
		if np.array(stateBatch).ndim == 1:
			stateBatch = np.array([stateBatch])
		graph = self.model.graph
		state_ = graph.get_collection_ref("inputs")[0]
		actionIndices_ = graph.get_collection_ref("actionIndices")[0]
		actionIndices = self.model.run(actionIndices_, feed_dict={state_: stateBatch})
		actionBatch = [self.actionSpace[i] for i in actionIndices]
		if len(actionBatch) == 1:
			actionBatch = actionBatch[0]
		return actionBatch


class ApproximateActionPrior:
	def __init__ (self, policyValueNet, actionSpace):
		self.policyValueNet = policyValueNet
		self.actionSpace = actionSpace

	def __call__(self, state):
		graph = self.policyValueNet.graph
		state_ = graph.get_collection_ref("inputs")[0]
		actionDist_ = graph.get_collection_ref("actionDist")[0]
		actionDist = self.policyValueNet.run(actionDist_, feed_dict={state_: np.array([state])})[0]
		actionPrior = {action: prob for action, prob in zip(self.actionSpace, actionDist)}
		return actionPrior


class ApproximateValueFunction:
	def __init__(self, policyValueNet):
		self.policyValueNet = policyValueNet

	def __call__(self, stateBatch):
		scalarOutput = False
		if np.array(stateBatch).ndim == 1:
			stateBatch = np.array([stateBatch])
			scalarOutput = True
		graph = self.policyValueNet.graph
		state_ = graph.get_collection_ref("inputs")[0]
		valuePrediction_ = graph.get_collection_ref("valuePrediction")[0]
		valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})
		if scalarOutput:
			valuePrediction = valuePrediction[0][0]
		return valuePrediction
