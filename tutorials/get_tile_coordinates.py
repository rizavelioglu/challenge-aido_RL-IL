from duckietown_rl.gym_duckietown.simulator import Simulator
from utils.helpers import SteeringToWheelVelWrapper
import cv2
import csv

# To convert [speed, steering] to wheel velocities: [leftWheelV, rightWheelV]
wrapper = SteeringToWheelVelWrapper()
# Create a list of environment names whose coordinate information will be collected
map_names = ["zigzag_dists", "4way", "loop_empty", "small_loop"]
# Initialize a dictionary where the info will be stored
tile_coords = {map_name: [] for map_name in map_names}

for map in map_names:
    env = Simulator(seed=123,
                    map_name=map,                   # Choose a map name to start with
                    max_steps=5000001,              # The max. # of steps can be taken before env. resets itself
                    domain_rand=True,               # If true, applies domain randomization
                    camera_width=640,               # Camera width for rendering
                    camera_height=480,              # Camera height for rendering
                    accept_start_angle_deg=4,       # The angle, in degrees, for the agent to turn to get aligned with the right lane's center
                    full_transparency=True,         # If true, makes available for all the env. info to be accessed
                    distortion=True,                # Distorts the image/observation so that it looks like in the real world. Points to sim-2-real problem
                    randomize_maps_on_reset=False,  # If true, maps are randomly chosen after each episode
                    draw_curve=False,               # Draws the right lane's center curve
                    draw_bbox=False)                # Renders the environment in top-down view

    # Reset environment for a fresh start & initialize variables
    obs = env.reset()
    # env.render()
    # We only run 1 episode each map and for 2000 steps. 2000 is enough for PID controller to finish the map
    EPISODES, STEPS = 1, 2000

    for episode in range(0, EPISODES):
        # Initialize the list of 10 previous angle errors
        prev_angles = [0] * 10
        # Initialize the previous angle value
        prev_angle = 0
        # Initialize a list where all the tile coordinates are stored
        _tile_coords = []
        for steps in range(0, STEPS):
            # Get the position of the agent relative to the center of the right lane
            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            # Get how far the agent is from the right lane's center
            distance_to_road_center = lane_pose.dist
            # Get the angle, in radians, that the agent should turn in order to be aligned with the right lane
            angle_from_straight_in_rads = lane_pose.angle_rad

            # Set PID parameters
            k_p, k_d, k_i = 17, 9, 0.1  # 33, 8, 0.1
            # Change how fast the agent should drive(speed) & PID parameters when the following conditions occur
            if -0.5 < lane_pose.angle_deg < 0.5:
                speed = 1
            elif -1 < lane_pose.angle_deg < 1:
                speed = 0.9
            elif -2 < lane_pose.angle_deg < 2:
                speed = 0.8
            elif -10 < lane_pose.angle_deg < 10:
                speed = 0.5
            else:
                speed = 0.3

            # Append the angle error to the list
            prev_angles.append(abs(prev_angle - lane_pose.angle_deg))
            # Remove the oldest error from the list
            prev_angles.pop(0)
            # Store the previous angle
            prev_angle = lane_pose.angle_deg

            # Calculate 'steering' value w.r.t. the PID parameters & the values gathered from the environment
            steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads + k_i * sum(prev_angles)
            # To convert [speed, steering] to wheel velocities: [leftWheelV, rightWheelV]
            action = wrapper.convert([speed, steering])
            # Apply the action and gather info
            obs, reward, done, info = env.step(action)
            # env.render()

            i, j = env.get_grid_coords(env.cur_pos)
            tile_coord = env._get_tile(i, j)['coords']

            # Do not add the tile coord. if it has already been added
            if tile_coord not in _tile_coords:
                _tile_coords.append(tile_coord)

            # Uncomment to run frame-by-frame
            # cv2.imshow("obs", obs)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break

        # Add tile coords. to dictionary, if it's not already been done before
        if tile_coords[env.map_name] != _tile_coords:
            tile_coords[env.map_name] = _tile_coords

        env.reset()

    print(map)

w = csv.writer(open("tile_coordinates.csv", "w"))
for key, val in tile_coords.items():
    w.writerow([key, val])
