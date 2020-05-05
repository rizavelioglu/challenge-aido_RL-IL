from duckietown_rl.gym_duckietown.simulator import Simulator
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

L2_LAMBDA = 1e-04


def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    return residual


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    nn = tf.keras.layers.add([rb_1, nn])

    nn = tf.layers.flatten(nn)

    return nn


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
                                kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                bias_initializer=tf.keras.initializers.he_normal(seed=self.seed))
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                bias_initializer=tf.keras.initializers.he_normal(seed=self.seed))

        model = tf.layers.dense(model, self._action.shape[1])

        return model

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self._observation)
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


env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=False, frame_skip=1, draw_DDPG_features=False)

OBSERVATIONS_SHAPE = (None, 480, 640, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/"
EPISODES = 10
STEPS = 1500
model = TfInference(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

observation = env.reset()
env.render()
cumulative_reward = 0.0
EPISODES = 10
STEPS = 65

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            env.reset()

        print(f"Reward: {reward:.2f}",
              f"\t| Action: [{action[0]:.3f}, {action[1]:.3f}]",
              f"\t| Speed: {env.speed:.2f}")

        cv2.imshow("obs", observation)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        env.render()
    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))

env.close()
model.close()
