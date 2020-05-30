import sys
sys.path.append("../../")
from duckietown_il.model_tf import one_residual
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


class TfInference:
    def __init__(self, observation_shape, action_shape, graph_location, seed=1234):
        # model definition
        self._observation = None
        self._action = None
        self._computation_graph = None
        self._optimizer = None

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

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 120)), self._observation)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize

    def _create(self, input_shape, output_shape):
        self._observation = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
        self._pre_process()

        self._computation_graph = self.computation_graph()

    def _storing(self, location):
        self.tf_saver = tf.train.Saver()

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            raise IOError('No model found...')

    def close(self):
        self.tf_session.close()

