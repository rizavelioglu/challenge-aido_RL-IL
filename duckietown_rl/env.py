def launch_env(id=None):
    from gym_duckietown.simulator import Simulator
    env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                    camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                    randomize_maps_on_reset=True, draw_curve=False, draw_bbox=False)

    return env
