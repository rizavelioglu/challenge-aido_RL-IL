import numpy as np
from duckietown_rl.ddpg import DDPG
from duckietown_rl.gym_duckietown.simulator import Simulator
from duckietown_rl.wrappers import ActionWrapper
import torch
import cv2

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=True)

# Action wrapper
# env = ActionWrapper(env)

state_dim = env.get_features().shape[0]    # @riza: state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="dense")
policy.load("model", directory="models", for_inference=True)

with torch.no_grad():
    while True:
        env.reset()        # obs = env.reset()
        obs = env.get_features()  # @riza
        env.render()
        rewards = []
        while True:
            action = policy.predict(np.array(obs))
            _, rew, done, misc = env.step(action)
            obs = env.get_features()  # @riza
            rewards.append(rew)
            env.render()

            print(f"Reward: {rew:.2f} | Action: {action}")
            # cv2.imshow("obs", obs)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break

            if done:
                break
        print("mean episode reward:", np.mean(rewards))
