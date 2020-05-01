from gym_duckietown.simulator import Simulator
import sys
import pyglet
from pyglet.window import key
import numpy as np
import pandas as pd

env = Simulator(seed=123, map_name="straight_road", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=False)

env.reset()
env.render()

data = []

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.ENTER:
        df_test = pd.DataFrame(data, columns=["step", "speed", "vel_left", "vel_right"])
        df_test.to_csv("get_dynamics_raw.csv")


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([1, 1])
    if key_handler[key.DOWN]:
        action = np.array([-1, -1])
    if key_handler[key.LEFT]:
        action = np.array([0.35, 1])
    if key_handler[key.RIGHT]:
        action = np.array([1, 0.35])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    obs, reward, done, info = env.step(action)

    data.append([env.unwrapped.step_count,
                 env.speed,
                 env.wheelVels[0],
                 env.wheelVels[1]])

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()
env.close()
