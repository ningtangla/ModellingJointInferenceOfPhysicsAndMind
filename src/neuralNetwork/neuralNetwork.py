import tensorflow as tf


class GeneratePolicyNet:
    def __init__(self, numStateSpace, numActionSpace, learningRate, regularizationFactor):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.learningRate = learningRate
        self.regularizationFactor = regularizationFactor

    def __call__(self, hiddenDepth, hiddenWidth, summaryPath="./tbdata"):
        tf.set_random_seed(128)

        print("Generating Policy Net with hidden layers: {} x {} = {}".format(hiddenDepth, hiddenWidth,
                                                                              hiddenDepth * hiddenWidth))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
                tf.add_to_collection("inputs", state_)
                tf.add_to_collection("inputs", actionLabel_)

            with tf.name_scope("hidden"):
                initWeight = tf.random_uniform_initializer(-0.03, 0.03)
                initBias = tf.constant_initializer(0.001)

                fc1 = tf.layers.Dense(units=hiddenWidth, activation=tf.nn.relu, kernel_initializer=initWeight,
                                      bias_initializer=initBias)
                a1_ = fc1(state_)
                w1_, b1_ = fc1.weights
                tf.summary.histogram("w1", w1_)
                tf.summary.histogram("b1", b1_)
                tf.summary.histogram("a1", a1_)

                a_ = a1_
                for i in range(2, hiddenDepth + 1):
                    fc = tf.layers.Dense(units=hiddenWidth, activation=tf.nn.relu, kernel_initializer=initWeight,
                                         bias_initializer=initBias)
                    aNext_ = fc(a_)
                    a_ = aNext_
                    w_, b_ = fc.weights
                    tf.summary.histogram("w{}".format(i), w_)
                    tf.summary.histogram("b{}".format(i), b_)
                    tf.summary.histogram("a{}".format(i), a_)

                fcLast = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                         bias_initializer=initBias)
                allActionActivation_ = fcLast(a_)
                wLast_, bLast_ = fcLast.weights
                tf.summary.histogram("wLast", wLast_)
                tf.summary.histogram("bLast", bLast_)
                tf.summary.histogram("allActionActivation", allActionActivation_)

            with tf.name_scope("outputs"):
                actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionDistribution_,
                                                                           labels=actionLabel_, name='cross_entropy')
                loss_ = tf.reduce_mean(cross_entropy, name='loss_')
                tf.add_to_collection("loss", loss_)
                lossSummary = tf.summary.scalar("loss", loss_)

                actionIndices_ = tf.argmax(actionDistribution_, axis=1)
                actionLabelIndices_ = tf.argmax(actionLabel_, axis=1)
                accuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, actionLabelIndices_), tf.float32))
                tf.add_to_collection("accuracy", accuracy_)
                accuracySummary = tf.summary.scalar("accuracy", accuracy_)

            with tf.name_scope("train"):
                l2RegularizationLoss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                                                  'bias' not in v.name]) * self.regularizationFactor

                optimizer = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_')
                gradVarPairs_ = optimizer.compute_gradients(loss_ + l2RegularizationLoss_)
                trainOp = optimizer.apply_gradients(gradVarPairs_)
                tf.add_to_collection(tf.GraphKeys.TRAIN_OP, trainOp)

                gradients_ = [tf.reshape(grad, [1, -1]) for (grad, _) in gradVarPairs_]
                gradTensor_ = tf.concat(gradients_, 1)
                gradNorm_ = tf.norm(gradTensor_)
                tf.add_to_collection("gradNorm", gradNorm_)
                tf.summary.histogram("gradients", gradTensor_)
                tf.summary.scalar('gradNorm', gradNorm_)

            fullSummary = tf.summary.merge_all()
            evalSummary = tf.summary.merge([lossSummary, accuracySummary])
            tf.add_to_collection("summaryOps", fullSummary)
            tf.add_to_collection("summaryOps", evalSummary)

            trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
            testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
            tf.add_to_collection("writers", trainWriter)
            tf.add_to_collection("writers", testWriter)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model
