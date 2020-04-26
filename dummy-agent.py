from duckietown_rl.gym_duckietown.simulator import Simulator
from utils.helpers import SteeringToWheelVelWrapper
import cv2
import numpy as np

# To convert to wheel velocities
wrapper = SteeringToWheelVelWrapper()
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=True, frame_skip=4, evaluate=True)

obs = env.reset()
env.render()
EPISODES, STEPS = 5, 2000

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):

        action = np.array([1, 1])
        obs, reward, done, info = env.step(action)

        try:
            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            dist_env = lane_pose.dist
        except:
            dist_env = 1.0

        features = env.get_features()
        env.render()

        print(f"Reward: {reward:.2f}",
              f"\t| Speed: {env.speed:.2f}",
              f"\t| Dist.: {abs(dist_env):.3f}",
              f"\t| Action: {action}")

        cv2.imshow("obs", obs)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        if done:
            break

    if env.env_count + 1 == 7:
        break

    env.reset()
