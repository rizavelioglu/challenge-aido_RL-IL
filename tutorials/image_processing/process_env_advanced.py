"""
Code based on
    - https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    - https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
"""
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from get_controller_params import get_params

env = DuckietownEnv(
    map_name="udem1",  # "zigzag_dists", "4way", "loop_empty","small_loop", "straight_road", "small_loop_cw"
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


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can
    # use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of
    # the frame frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top
    # the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(-240, height), (840, height), (320, 100)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment


def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    
    if lines is not None:
        # Loops through every detected line
        for line in lines:
            # Reshapes line from 2D array to 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe
            # the slope and y-intercept
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the
            # lane
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))

        if left:
            left_avg = np.average(left, axis=0)
            left_line = calculate_coordinates(frame, left_avg)
            
            if not right:
                return np.array([left_line, np.zeros_like(left_line)])
        
        if right:
            right_avg = np.average(right, axis=0)
            right_line = calculate_coordinates(frame, right_avg)
    
            if not left:
                return np.array([np.zeros_like(right_line), right_line])

        # Averages out all the values for left and right into a single slope and y-intercept value for each line
        left_avg = np.average(left, axis=0)
        right_avg = np.average(right, axis=0)
    
        # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
        left_line = calculate_coordinates(frame, left_avg)
        right_line = calculate_coordinates(frame, right_avg)

        return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)

    if slope != 0:
    
        # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
        x1 = int((y1 - intercept) / slope)
        # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
        x2 = int((y2 - intercept) / slope)
    else:
        x1 = x2 = 0
    return np.array([x1, y1, x2, y2])


def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            if -100000 < x1 < 100000 and -100000 < x2 < 100000:
                cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return lines_visualize


def img_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = auto_canny(blurred)
    segmented = do_segment(canny)
    cv2.imshow("Segmented", segmented)
    
    hough = cv2.HoughLinesP(segmented, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
    # Averages multiple detected lines from hough into one
    # line for left border of lane and one line for right border of lane
    lines = calculate_lines(img, hough)
    # Visualizes the lines
    lines_visualize = visualize_lines(img, lines)
    cv2.imshow("hough", lines_visualize)
    # Overlays lines on frame by taking their weighted sums and adding an
    # arbitrary scalar value of 1 as the gamma argument
    out = cv2.addWeighted(img, 0.9, lines_visualize, 1, 1)
    return out


while True:
    # Get the action from the PID controller
    action = get_params(env)
    # Take the action in the environment & get the next observation, reward, etc.
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Render the env.
    env.render()
    # Apply image processing to the observations
    output = img_processing(obs)
    # Opens a new window and displays the output frame
    cv2.imshow('Output', output)
    # Close all the windows when "q" is pressed on keyboard
    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
