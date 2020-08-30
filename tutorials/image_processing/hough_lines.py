"""
Simple exercise to construct a PID controller that controls the simulated Duckiebot using pose and rendering Detected
Lane Lines using Canny Edge Detection and Hough Lines.
"""
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2
import numpy as np
from get_controller_params import get_params

env = DuckietownEnv(
    map_name="udem1",  # "zigzag_dists", "4way", "loop_empty","small_loop", "straight_road", "small_loop_cw"
    domain_rand=True,
    distortion=False,
    max_steps=1500,
    draw_curve=False,
    draw_bbox=False)
obs = env.reset()
env.render()
total_reward = 0

while True:
    # Get the action from the PID controller
    action = get_params(env)
    # Take the action in the environment & get the next observation, reward, etc.
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Render the env.
    env.render()
    # Edge Detection
    dst = cv2.Canny(obs, 200, 225, None, 3)
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    
    # Source Code: https://docs.opencv.org/4.1.0/d9/db0/tutorial_hough_lines.html
    # Probabilistic Line Transform
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    # Show results
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # Close all the windows when "q" is pressed on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
