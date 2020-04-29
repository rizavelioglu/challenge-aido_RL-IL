"""
This script outputs a 'good' starting position and angle for an agent, given a map. That means, agent starts at a point
that is close to the center of a lane and starts at an angle that is close to zero, which means agent is aligned with
the lane.
"""

from duckietown_world.world_duckietown.sampling_poses import sample_good_starting_pose
import duckietown_world as dw
import geometry as geo
import numpy as np

along_lane = 1
only_straight = True
m = dw.load_map("zigzag_dists")
q = sample_good_starting_pose(m, only_straight=only_straight, along_lane=along_lane)

translation, angle = geo.translation_angle_from_SE2(q)
propose_pos = np.array([translation[0], 0, translation[1]])
propose_angle = angle

print(f"Pose: {propose_pos}",
      f"\t|\tAngle: {propose_angle}")
