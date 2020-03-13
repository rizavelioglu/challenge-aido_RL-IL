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

print(f"pose: {propose_pos}",
      f"\nangle: {propose_angle}")
