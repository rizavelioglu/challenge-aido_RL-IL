"""
Explanation on how Canny Edge Detector works:
    https://www.youtube.com/watch?v=5dL7FvL-oy0
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load an RGB image on a gray scale
img = cv2.imread("../images/road.png", 0)
# One way to get rid of the noise on the image, is by applying Gaussian blur to smooth it. To do so, image convolution
# technique is applied with a Gaussian Kernel (3x3, 5x5, 7x7 etc…). The kernel size depends on the expected blurring
# effect. Basically, the smallest the kernel, the less visible is the blur.
edges = cv2.Canny(img,
                  threshold1=100,
                  threshold2=200,
                  apertureSize=3,   # size for the Sobel operator. 
                  L2gradient=True)  # a flag, indicating whether a more accurate L2 norm should be used to calculate the
                                    # image grad magnitude or whether the default L1 norm

# Show all the images
cv2.imshow("Original Image-Grayscaled", img)
cv2.imshow("Edge Image", edges)

# Close all the windows when "q" is pressed on keyboard
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# =============================================================================
# =============================== MASKING =====================================
# =============================================================================
# Load the image
img = cv2.imread("../images/road.png")
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

# Add all the masks
final_mask = yellow + white + red
# Remove everything except the mask from the image
target = cv2.bitwise_and(img, img, mask=final_mask)
# Blur the image to remove false detections
blur = cv2.GaussianBlur(target, (7, 7), 0)

# Show all the images
cv2.imshow("Original Image", img)
cv2.imshow("Mixed", target)
cv2.imshow("Blurred", blur)

# Close all the windows when "q" is pressed on keyboard
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# =============================================================================
# =============================== AUTO CANNY  =================================
# =============================================================================
"""
Originally taken from:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


# Convert the blurred image to gray scale
gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide_blur = cv2.Canny(gray_blur, 10, 200)
tight_blur = cv2.Canny(gray_blur, 225, 250)
auto_blur = auto_canny(gray_blur)

# Now do the same for the un-blurred image, just to see the difference whether blurring works or not
# Convert the blurred image to gray scale
gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(gray_target, 10, 200)
tight = cv2.Canny(gray_target, 225, 250)
auto = auto_canny(gray_target)

# show both images
cv2.imshow("Original", target)
cv2.imshow("Edges w/ Blurred Image", np.hstack([wide_blur, tight_blur, auto_blur]))
cv2.imshow("Edges w/o Blurred Image", np.hstack([wide, tight, auto]))

# Close all the windows when "q" is pressed on keyboard
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# =============================================================================
# =============================== Brightening Image ===========================
# =============================================================================
"""
Description and usage explained here:
    https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
"""
alpha = 1.5    # Simple contrast control [1.0-3.0]
beta = 50     # Simple brightness control [0-100]

new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

cv2.imshow('Original Image', img)
cv2.imshow('New Image', new_image)
# Wait until user press some key
cv2.waitKey()

# Blur the image
blur = cv2.GaussianBlur(new_image, (7, 7), 0)
# Convert blurred im. to grayscale
gray_im = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# Convert the im. to grayscale
im = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide_blur = cv2.Canny(gray_im, 10, 200)
tight_blur = cv2.Canny(gray_im, 225, 250)
auto_blur = auto_canny(gray_im)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(im, 10, 200)
tight = cv2.Canny(im, 225, 250)
auto = auto_canny(im)

# Show both images
cv2.imshow("Edges w/ Blurred Image", np.hstack([wide_blur, tight_blur, auto_blur]))
cv2.imshow("Edges w/o Blurred Image", np.hstack([wide, tight, auto]))

# Close all the windows when "q" is pressed on keyboard
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# =============================================================================
# =============================== Gamma Correction ===========================
# =============================================================================
gamma = 0.4

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res_org = cv2.LUT(img, lookUpTable)


lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res_br = cv2.LUT(new_image, lookUpTable)


cv2.imshow("Original Image", img)
cv2.imshow("Brightened Image", new_image)
cv2.imshow("Org. Image w/ Gamma Correction (gamma=0.4)", res_org)
cv2.imshow("Brightened Image w/ Gamma Correction (gamma=0.4)", res_br)

# Close all the windows when "q" is pressed on keyboard
if cv2.waitKey() & 0xFF == ord('q'):
    cv2.destroyAllWindows()
