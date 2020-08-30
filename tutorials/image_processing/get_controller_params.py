"""
This script sets up a PID controller and its parameters (very similar to the one in '/tutorials/PIDcontroller.py') and
returns the action to be taken by the environment.
"""


def get_params(env):
    # Get the position of the agent relative to the center of the right lane
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    # Get how far the agent is from the right lane's center
    distance_to_road_center = lane_pose.dist
    # Get the angle, in radians, that the agent should turn in order to be aligned with the right lane
    angle_from_straight_in_rads = lane_pose.angle_rad

    # Change how fast the agent should drive(speed) & PID parameters when the following conditions occur
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    k_p, k_d = 17, 9
    if -0.5 < lane_pose.angle_deg < 0.5:
        speed = 0.9
    elif -1 < lane_pose.angle_deg < 1:
        speed = 0.8
    elif -10 < lane_pose.angle_deg < 10:
        k_p, k_d = 33, 8
        speed = 0.4
    else:
        k_p, k_d = 33, 8
        speed = 0.3

    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads
    # Action to be passed to the environment
    action = [speed, steering]
    return action
