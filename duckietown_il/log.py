import sys
sys.path.append("../")
import time
import cv2
import torch
import numpy as np
from duckietown_rl.gym_duckietown.simulator import Simulator
from duckietown_rl.ddpg import DDPG
from _loggers import Logger

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=True, draw_curve=False, draw_bbox=False, frame_skip=4, draw_DDPG_features=False)

state_dim = env.get_features().shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
expert = DDPG(state_dim, action_dim, max_action, net_type="dense")
expert.load("model", directory="../duckietown_rl/models", for_inference=True)

# Initialize the environment
env.reset()
# Get features(state representation) for RL agent
obs = env.get_features()
EPISODES, STEPS = 400, 256
DEBUG = False

# please notice
logger = Logger(env, log_file=f'train-{int(EPISODES*STEPS/1000)}k.log')

start_time = time.time()
print(f"[INFO]Starting to get logs for {EPISODES} episodes each {STEPS} steps..")
with torch.no_grad():
    # let's collect our samples
    for episode in range(0, EPISODES):
        for steps in range(0, STEPS):
            # we use our 'expert' to predict the next action.
            action = expert.predict(np.array(obs))
            # Apply the action
            observation, reward, done, info = env.step(action)
            # Get features(state representation) for RL agent
            obs = env.get_features()

            if done:
                print(f"#Episode: {episode}\t | #Step: {steps}")
                break

            closest_point, _ = env.closest_curve_point(env.cur_pos, env.cur_angle)
            if closest_point is None:
                done = True
                break
            # Cut the horizon: obs.shape = (480,640,3) --> (300,640,3)
            observation = observation[150:450, :]
            # we can resize the image here
            observation = cv2.resize(observation, (120, 60))
            # NOTICE: OpenCV changes the order of the channels !!!
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

            # we may use this to debug our expert.
            if DEBUG:
                cv2.imshow('debug', observation)
                cv2.waitKey(1)

            logger.log(observation, action, reward, done, info)

        logger.on_episode_done()  # speed up logging by flushing the file
        env.reset()

logger.close()
env.close()

end_time = time.time()
print(f"Process finished. It took {(end_time - start_time) / (60*60):.2f} hours!")
