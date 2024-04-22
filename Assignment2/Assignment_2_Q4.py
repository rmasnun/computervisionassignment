import cv2
import numpy as np

def compute_integral_image(img):
    """
    Compute the integral image of a grayscale image.

    Parameters:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Integral image.
    """
    integral_img = np.zeros_like(img, dtype=np.uint32)

    # Compute the first row and first column of the integral image
    integral_img[0, 0] = img[0, 0]

    for i in range(1, img.shape[0]):
        integral_img[i, 0] = integral_img[i - 1, 0] + img[i, 0]

    for j in range(1, img.shape[1]):
        integral_img[0, j] = integral_img[0, j - 1] + img[0, j]

    # Compute the rest of the integral image
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            integral_img[i, j] = img[i, j] + integral_img[i - 1, j] + integral_img[i, j - 1] - integral_img[i - 1, j - 1]

    return integral_img

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute integral image
    integral_img = compute_integral_image(gray)

    # Display RGB feed
    cv2.imshow('RGB Feed', frame)

    # Display integral image
    cv2.imshow('Integral Image', integral_img.astype(np.uint8))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
