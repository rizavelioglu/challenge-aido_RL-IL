"""
Taken from:
    https://www.hackster.io/kemfic/simple-lane-detection-c3db2f

better version:
    https://www.hackster.io/kemfic/curved-lane-detection-34f771
"""
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2
import numpy as np
from get_controller_params import get_params

env = DuckietownEnv(
    map_name="udem1",     # "zigzag_dists", "4way", "loop_empty","small_loop", "straight_road", "small_loop_cw"
    domain_rand=True,
    distortion=False,
    max_steps=3000,
    draw_curve=False,
    draw_bbox=False,
    accept_start_angle_deg=30)
obs = env.reset()
env.render()

total_reward = 0


def color_filter(image):
    # convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0, 150, 0])                 # lower = np.array([0,190,0])
    upper = np.array([255, 255, 255])
    yellower = np.array([10, 0, 90])
    yelupper = np.array([50, 255, 255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(-200), int(y)], [int(x+200), int(y)], [int(0.55*x), int(0.1*y)], [int(0.45*x), int(0.1*y)]])
    # define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)
    # Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    # returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


rightSlope, leftSlope, rightIntercept, leftIntercept = [], [], [], []


def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor = [0, 255, 0]
    leftColor = [255, 0, 0]
    
    # this is used to filter out the outlying lines that can affect the average
    # We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500:
                    yintercept = y2 - (slope*x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
    # We use slicing operators and np.mean() to find the averages of the 30 previous frames
    # This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    # Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 0, 255))
        cv2.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
        # I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def linedetect(img):
    return hough_lines(img, 1, np.pi/180, 10, 20, 100)


while True:
    # Get the action from the PID controller
    action = get_params(env)
    # Take the action in the environment & get the next observation, reward, etc.
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Render the env.
    env.render()
    
    masked = color_filter(obs)
    segmented = roi(masked)
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 50, 120)
    hough_img = linedetect(canny)

    # Show all the processed images & all of them combined
    cv2.imshow("img", masked)
    cv2.imshow("segmented", segmented)
    cv2.imshow("Canny", canny)
    cv2.imshow("Hough", hough_img)
    cv2.imshow("out", cv2.addWeighted(obs, 1, hough_img, 0.8, 0))
    
    # Close all the windows when "q" is pressed on keyboard
    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
