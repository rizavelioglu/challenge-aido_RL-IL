import sys
sys.path.append("../")
from duckietown_rl.gym_duckietown.simulator import Simulator
import cv2
import numpy as np
from model_tf import one_residual
from submission.tf.model import TfInference
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# Load model
OBSERVATIONS_SHAPE = (None, 480, 640, 3)   # here we assume the observations have been resized to 60x80
ACTIONS_SHAPE = (None, 2)                  # actions have a shape of 2: [leftWheelVelocity, rightWheelVelocity]
SEED = 1234
STORAGE_LOCATION = "trained_models/"       # where we store our trained models

model = TfInference(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,            # same
    graph_location=STORAGE_LOCATION,       # where do we want to store our trained models
    seed=SEED)                             # to seed all random operations in the model (e.g., dropout)

# Load the environment
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=False, frame_skip=1, draw_DDPG_features=False)

observation = env.reset()
env.render()
cumulative_reward = 0.0

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
