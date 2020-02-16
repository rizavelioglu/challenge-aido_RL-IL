import numpy as np
from model import DDPG
from gym_duckietown.simulator import Simulator
from utils.helpers import SteeringToWheelVelWrapper
import cv2

# To convert to wheel velocities
wrapper = SteeringToWheelVelWrapper()
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=True, draw_curve=False, draw_bbox=True, user_tile_start=(2, 1))

np.random.seed(123)
model = DDPG(state_dim=14, action_dim=2, max_action=1, net_type="dense")
model.load("model", "models", for_inference=True)

env.render()
EPISODES, STEPS = 100, 2000

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):

        features = env.get_features()

        try:
            action = model.predict(features)
        except:
            env.reset()

        obs, reward, done, info = env.step(action.astype(float))
        env.render()


        print(reward)

        # cv2.imshow("obs", obs)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

    env.reset()
