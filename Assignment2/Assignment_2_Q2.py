import cv2
import numpy as np

# Load the video footage
cap = cv2.VideoCapture('v.mp4')  # Replace 'video_footage.mp4' with the path to your video

# Select a frame from the video
frame_number = 1000  # Change this to the desired frame number
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Compute gradients using Sobel operators
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Compute elements of the structure tensor
grad_xx = grad_x ** 2
grad_xy = grad_x * grad_y
grad_yy = grad_y ** 2

# Apply Gaussian blur to the structure tensor elements
ksize = 3  # Kernel size for Gaussian blur
sigma = 1.0  # Standard deviation for Gaussian blur
grad_xx_blurred = cv2.GaussianBlur(grad_xx, (ksize, ksize), sigma)
grad_xy_blurred = cv2.GaussianBlur(grad_xy, (ksize, ksize), sigma)
grad_yy_blurred = cv2.GaussianBlur(grad_yy, (ksize, ksize), sigma)

# Compute the Harris response function
k = 0.04  # Harris parameter
harris_response = (grad_xx_blurred * grad_yy_blurred - grad_xy_blurred ** 2) - k * (grad_xx_blurred + grad_yy_blurred) ** 2

# Apply thresholding to identify corner candidates
threshold = 0.01 * np.max(harris_response)
corner_candidates = np.zeros_like(gray)
corner_candidates[harris_response > threshold] = 255

# Perform non-maximum suppression to select the strongest corners
corners = cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)
corners = cv2.dilate(corners, None)

# Display the original frame and detected corners
cv2.imshow('Original Frame', frame)
cv2.imshow('Detected Corners', corners)
cv2.waitKey(0)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
