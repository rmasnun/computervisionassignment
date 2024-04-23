import cv2
import numpy as np

# Load the video footage
cap = cv2.VideoCapture('v.mp4')  # Replace 'video_footage.mp4' with the path to your video

# Select a frame from the video
frame_number = 1000  # Change this to the desired frame number
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, img = cap.read()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform corner detection using Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate the corner points to make them clearer
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Display the result
cv2.imshow('Corners Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()