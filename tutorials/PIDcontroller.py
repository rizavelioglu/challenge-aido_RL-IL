import sys
sys.path.append("../")
from duckietown_rl.gym_duckietown.simulator import Simulator
from tutorials.helpers import SteeringToWheelVelWrapper

# To convert [speed, steering] to wheel velocities: [leftWheelV, rightWheelV]
wrapper = SteeringToWheelVelWrapper()
# Create the environment with the specified parameters
env = Simulator(seed=123,
                map_name="zigzag_dists",        # Choose a map name to start with
                max_steps=5000001,              # The max. # of steps can be taken before env. resets itself
                domain_rand=True,               # If true, applies domain randomization
                camera_width=640,               # Camera width for rendering
                camera_height=480,              # Camera height for rendering
                accept_start_angle_deg=4,       # The angle, in degrees, for the agent to turn to get aligned with the right lane's center
                full_transparency=True,         # If true, makes available for all the env. info to be accessed
                distortion=True,                # Distorts the image/observation so that it looks like in the real world. Points to sim-2-real problem
                randomize_maps_on_reset=True,   # If true, maps are randomly chosen after each episode
                draw_curve=False,               # Draws the right lane's center curve
                draw_bbox=True)                 # Renders the environment in top-down view

# Reset environment for a fresh start & initialize variables
obs = env.reset()
# Render environment
env.render()
# Specify how long the PID controller should be run
EPISODES, STEPS = 100, 800

for episode in range(0, EPISODES):
    # Initialize the list of 10 previous angle errors, which will be used for the I parameter of PID controller
    prev_angles = [0] * 10
    # Initialize the previous angle value
    prev_angle = 0

    for steps in range(0, STEPS):
        # Get the position of the agent relative to the center of the right lane
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        # Get how far the agent is from the right lane's center
        distance_to_road_center = lane_pose.dist
        # Get the angle, in radians, that the agent should turn in order to be aligned with the right lane
        angle_from_straight_in_rads = lane_pose.angle_rad

        # Set PID parameters
        k_p, k_d, k_i = 17, 9, 0.1
        # Change how fast the agent should drive(speed) & PID parameters when the following conditions occur
        if -0.5 < lane_pose.angle_deg < 0.5:
            speed = 1
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
        # Render environment
        env.render()
        # Print info
        print(f"Reward: {reward:.2f}",
              f"\t| Action: [{action[0]:.3f}, {action[1]:.3f}]",
              f"\t| Speed: {env.speed:.2f}",
              f"\t| Dist.: {distance_to_road_center:.3f}")

        # Uncomment to run frame-by-frame
        # cv2.imshow("obs", obs)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

    # Reset the environment and the variables after each episode
    env.reset()
