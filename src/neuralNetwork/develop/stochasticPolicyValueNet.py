import tensorflow as tf
import numpy as np
import dataTools


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

				a_ = a1_
				for i in range(1, len(hiddenWidths)-1):
					fc = tf.layers.Dense(units=hiddenWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
					aNext_ = fc(a_)
					a_ = aNext_

				fcAction = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aAction_ = fcAction(a_)
				fcLastAction = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				allActionActivation_ = fcLastAction(aAction_)

				fcValue = tf.layers.Dense(units=hiddenWidths[-1], activation=tf.nn.relu, kernel_initializer=initWeight, bias_initializer=initBias)
				aValue_ = fcValue(a_)
				fcLastValue = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight, bias_initializer=initBias)
				valueActivation_ = fcLastValue(aValue_)

			with tf.name_scope("outputs"):
				actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_, labels=actionLabel_, name='cross_entropy')
				actionLoss_ = tf.reduce_mean(cross_entropy, name='actionLoss_')
				tf.add_to_collection("actionLoss", actionLoss_)
				actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

				actionIndices_ = tf.random.categorical(actionDistribution_, 1)
				tf.add_to_collection("actionIndices", actionIndices_)

				valuePrediction_ = valueActivation_
				valueLoss_ = tf.sqrt(tf.losses.mean_squared_error(valueLabel_, valuePrediction_))
				tf.add_to_collection("valuePrediction", valuePrediction_)
				tf.add_to_collection("valueLoss", valueLoss_)
				valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

				relativeErrorBound_ = tf.constant(self.valueRelativeErrBound)
				relativeValueError_ = tf.abs((valuePrediction_ - valueLabel_) / valueLabel_)
				valueAccuracy_ = tf.reduce_mean(tf.cast(tf.less(relativeValueError_, relativeErrorBound_), tf.float32))
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
			evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary, valueAccuracySummary])
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
		state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
		actionLossCoef_, valueLossCoef_ = graph.get_collection_ref("lossCoefs")
		loss_ = graph.get_collection_ref("loss")[0]
		actionLoss_ = graph.get_collection_ref("actionLoss")[0]
		valueLoss_ = graph.get_collection_ref("valueLoss")[0]
		valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
		trainOp = graph.get_collection_ref(tf.GraphKeys.TRAIN_OP)[0]
		fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
		trainWriter = graph.get_collection_ref('writers')[0]
		fetches = [{"loss": loss_, "actionLoss": actionLoss_, "valueLoss": valueLoss_, "valueAcc": valueAccuracy_},
				   trainOp, fullSummaryOp]

		stateBatch, actionLabelBatch, valueLabelBatch = trainingData

		lossHistory = np.ones(self.lossHistorySize)
		valueAccuracyHistory = np.zeros(self.lossHistorySize)
		actionLossCoef = 30
		valueLossCoef = 1
		coefUpdated = False

		for stepNum in range(self.maxStepNum):
			evalDict, _, summary = model.run(fetches, feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch, valueLabel_: valueLabelBatch,
																 actionLossCoef_: actionLossCoef, valueLossCoef_: valueLossCoef})

			if stepNum % self.reportInterval == 0 and not coefUpdated:
				# actionLossCoef = evalDict["valueLoss"] / evalDict["actionLoss"]
				if evalDict["actionLoss"] < 0.6:
					actionLossCoef = 5
					valueLossCoef = 1
					coefUpdated = True
					print("Coefficients of losses Updated to {:.2f} {:.2f}".format(actionLossCoef, valueLossCoef))

			if self.summaryOn and (stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum-1):
				trainWriter.add_summary(summary, stepNum)
				evaluate(model, self.testData, summaryOn=True, stepNum=stepNum)

			if stepNum % self.reportInterval == 0:
				print("#{} {}".format(stepNum, evalDict))

			lossHistory[stepNum % self.lossHistorySize] = evalDict["loss"]
			lossChange = np.mean(np.abs(lossHistory - np.min(lossHistory)))
			valueAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["valueAcc"]

			if lossChange < self.lossChangeThreshold:
				break

		return model


def evaluate(model, testData, summaryOn=False, stepNum=None):
	graph = model.graph
	state_, actionLabel_, valueLabel_ = graph.get_collection_ref("inputs")
	loss_ = graph.get_collection_ref("loss")[0]
	actionLoss_ = graph.get_collection_ref("actionLoss")[0]
	valueLoss_ = graph.get_collection_ref("valueLoss")[0]
	valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
	evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
	testWriter = graph.get_collection_ref('writers')[1]
	fetches = [{"actionLoss": actionLoss_, "valueLoss": valueLoss_, "valueAcc": valueAccuracy_},
			   evalSummaryOp]

	stateBatch, actionLabelBatch, valueLabelBatch = testData
	evalDict, summary = model.run(fetches, feed_dict={state_: stateBatch, actionLabel_: actionLabelBatch, valueLabel_: valueLabelBatch})
	if summaryOn:
		testWriter.add_summary(summary, stepNum)
	return evalDict


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


def approximatePolicy(stateBatch, policyValueNet, actionSpace):
	if np.array(stateBatch).ndim == 1:
		stateBatch = np.array([stateBatch])
	graph = policyValueNet.graph
	state_ = graph.get_collection_ref("inputs")[0]
	actionIndices_ = graph.get_collection_ref("actionIndices")[0]
	actionIndices = policyValueNet.run(actionIndices_, feed_dict={state_: stateBatch})
	actionBatch = [actionSpace[indexList[0]] for indexList in actionIndices]
	if len(actionBatch) == 1:
		actionBatch = actionBatch[0]
	return actionBatch


def approximateValueFunction(stateBatch, policyValueNet):
	# if np.array(stateBatch).ndim == 1:
	# 	stateBatch = np.array([stateBatch])
	graph = policyValueNet.graph
	state_ = graph.get_collection_ref("inputs")[0]
	valuePrediction_ = graph.get_collection_ref("valuePrediction")[0]
	valuePrediction = policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})
	# if len(valuePrediction) == 1:
	# 	valuePrediction = valuePrediction[0][0]
	return valuePrediction
