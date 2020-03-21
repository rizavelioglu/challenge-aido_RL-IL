from gym_duckietown.simulator import Simulator
from utils.helpers import SteeringToWheelVelWrapper
import cv2

# To convert to wheel velocities
wrapper = SteeringToWheelVelWrapper()
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=True)

obs = env.reset()
env.render()
EPISODES, STEPS = 100, 800

for episode in range(0, EPISODES):
    prev_angles = [0] * 10
    prev_angle = 0

    for steps in range(0, STEPS):
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        k_p, k_d, k_i = 17, 9, 0.1
        if -0.5 < lane_pose.angle_deg < 0.5:
            speed = 1
        elif -1 < lane_pose.angle_deg < 1:
            speed = 0.9
        elif -2 < lane_pose.angle_deg < 2:
            speed = 0.8
        elif -10 < lane_pose.angle_deg < 10:
            k_p, k_d = 33, 8
            speed = 0.5
        else:
            k_p, k_d, k_i = 33, 8, 0.05
            speed = 0.3

        prev_angles.append(abs(prev_angle - lane_pose.angle_deg))
        prev_angles.pop(0)
        prev_angle = lane_pose.angle_deg

        steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads + k_i * sum(prev_angles)
        action = wrapper.convert([speed, steering])
        obs, reward, done, info = env.step(action)

        features = env.get_features()
        env.render()

        # cv2.imshow("obs", obs)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

    env.reset()
