import numpy as np
import tensorflow.compat.v1 as tf

l2_lambda = 1e-04


def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    return residual


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    nn = tf.keras.layers.add([rb_1, nn])

    nn = tf.layers.flatten(nn)

    return nn


class TensorflowModel:
    def __init__(self, observation_shape, action_shape, graph_location, seed=1234):
        # model definition
        self._observation = None
        self._action = None
        self._computation_graph = None
        self._optimization_op = None

        self.tf_session = tf.InteractiveSession()

        # restoring
        self.tf_checkpoint = None
        self.tf_saver = None

        self.seed = seed

        self._initialize(observation_shape, action_shape, graph_location)

    def predict(self, state):
        action = self.tf_session.run(self._computation_graph, feed_dict={
            self._observation: [state],
        })
        return np.squeeze(action)

    def train(self, observations, actions):
        _, loss = self.tf_session.run([self._optimization_op, self._loss], feed_dict={
            self._observation: observations,
            self._action: actions
        })
        return loss

    def commit(self):
        self.tf_saver.save(self.tf_session, self.tf_checkpoint)

    def computation_graph(self):
        model = one_residual(self._preprocessed_state, seed=self.seed)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.initializers.he_normal(seed=self.seed),
                                bias_initializer=tf.initializers.he_normal(seed=self.seed))
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.initializers.he_normal(seed=self.seed),
                                bias_initializer=tf.initializers.he_normal(seed=self.seed))

        model = tf.layers.dense(model, self._action.shape[1])

        return model

    def _optimizer(self):
        return tf.train.AdamOptimizer()

    def _loss_function(self):
        return tf.losses.mean_squared_error(self._action, self._computation_graph)

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)
            self.tf_session.run(tf.global_variables_initializer())

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 120)), self._observation)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize

    def _create(self, input_shape, output_shape):
        self._observation = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
        self._pre_process()

        self._computation_graph = self.computation_graph()
        self._loss = self._loss_function()
        self._optimization_op = self._optimizer().minimize(self._loss)

    def _storing(self, location):
        self.tf_saver = tf.train.Saver()

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            self.tf_checkpoint = location

    def close(self):
        self.tf_session.close()
