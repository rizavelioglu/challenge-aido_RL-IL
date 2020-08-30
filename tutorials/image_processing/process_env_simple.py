"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose & visualizing processed images
"""
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2
import numpy as np
from get_controller_params import get_params

env = DuckietownEnv(
    map_name="udem1",  # "zigzag_dists", "4way", "loop_empty","small_loop", "small_loop_cw", "regress_4way_adam"
    domain_rand=True,
    distortion=False,
    max_steps=3000,
    draw_curve=False,
    draw_bbox=False)
obs = env.reset()
env.render()

total_reward = 0


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def brighten(img):
    alpha = 1.5    # Simple contrast control [1.0-3.0]
    beta = 50     # Simple brightness control [0-100]

    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_image


def masking(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # defining the range of yellow color
    yellow_lower = np.array([25, 120, 50], np.uint8)
    yellow_upper = np.array([60, 255, 198], np.uint8)
    
    # defining the range of white color
    white_lower = np.array([0, 0, 0], np.uint8)
    white_upper = np.array([0, 0, 255], np.uint8)
    
    # defining the range of red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    
    # finding the range of yellow & white color in the image
    yellow = cv2.inRange(img_HSV, yellow_lower, yellow_upper)
    white = cv2.inRange(img_HSV, white_lower, white_upper)
    red = cv2.inRange(img_HSV, red_lower, red_upper)
    
    final_mask = yellow + white + red
    
    target = cv2.bitwise_and(img, img, mask=final_mask)
    
    # Blur the image to remove false detections
    blur = cv2.GaussianBlur(target, (7, 7), 0)

    return blur


def img_processing(img):
    # Brighten the image
    img = brighten(img)
    # cv2.imshow("Brightened", brightened)
    # Apply color filtering/masking
    img = masking(img)
    # cv2.imshow("Masked", masked)
    # Blur the image
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # Return the processed image after applying canny edge detection
    return auto_canny(img)


while True:
    # Get the action from the PID controller
    action = get_params(env)
    # Take the action in the environment & get the next observation, reward, etc.
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Render the env.
    env.render()
    # Apply image processing to the observations
    dst = img_processing(obs)
    # Show the processed image
    cv2.imshow("img", dst)
    # Close all the windows when "q" is pressed on keyboard
    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
