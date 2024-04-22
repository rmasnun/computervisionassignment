import cv2
import numpy as np

# Load the video footage
cap = cv2.VideoCapture('v.mp4')  # Replace 'video_footage.mp4' with the path to your video

# Select a frame from the video
frame_number = 1000  # Change this to the desired frame number
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

print(frame)
# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Compute gradients using Sobel operators
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude and direction
grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
grad_dir = np.arctan2(grad_y, grad_x) * (180 / np.pi)

# Non-maximum suppression
edges = np.zeros_like(grad_mag)
rows, cols = edges.shape

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        angle = grad_dir[i, j]
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
            if grad_mag[i, j] >= grad_mag[i, j - 1] and grad_mag[i, j] >= grad_mag[i, j + 1]:
                edges[i, j] = grad_mag[i, j]
        elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
            if grad_mag[i, j] >= grad_mag[i - 1, j - 1] and grad_mag[i, j] >= grad_mag[i + 1, j + 1]:
                edges[i, j] = grad_mag[i, j]
        elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
            if grad_mag[i, j] >= grad_mag[i - 1, j] and grad_mag[i, j] >= grad_mag[i + 1, j]:
                edges[i, j] = grad_mag[i, j]
        elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
            if grad_mag[i, j] >= grad_mag[i - 1, j + 1] and grad_mag[i, j] >= grad_mag[i + 1, j - 1]:
                edges[i, j] = grad_mag[i, j]

# Apply double thresholding for edge tracing by hysteresis
high_threshold = 100  # Change this to adjust the high threshold
low_threshold = 50   # Change this to adjust the low threshold

strong_edges = (edges > high_threshold).astype(np.uint8) * 255
weak_edges = ((edges >= low_threshold) & (edges <= high_threshold)).astype(np.uint8) * 255

# Perform edge tracing using hysteresis
_, strong_edges = cv2.threshold(strong_edges, 0, 255, cv2.THRESH_BINARY)
_, weak_edges = cv2.threshold(weak_edges, 0, 255, cv2.THRESH_BINARY)

# Find contours of strong edges
contours, _ = cv2.findContours(strong_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original frame
frame_with_edges = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

# Display the original frame and the frame with detected edges
cv2.imshow('Original Frame', frame)
cv2.imshow('Frame with Detected Edges', frame_with_edges)
cv2.waitKey(0)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
