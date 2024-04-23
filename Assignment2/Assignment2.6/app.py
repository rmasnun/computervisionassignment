import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

def compute_integral_image(img):
    """
    Compute the integral image of a grayscale image.

    Parameters:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Integral image.
    """
    integral_img = np.zeros_like(img, dtype=np.uint8)

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

def find_keypoints_and_descriptors(image):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    # Check if either descriptor is None
    if descriptors1 is None or descriptors2 is None:
        return []

    # Ensure both descriptors have the same data type and number of columns
    descriptors1 = descriptors1.astype(np.uint8)
    descriptors2 = descriptors2.astype(np.uint8)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    # Draw matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def image_stitch(image1, image2):
    # Find keypoints and descriptors for both images
    keypoints1, descriptors1 = find_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = find_keypoints_and_descriptors(image2)

    # Match features between the two images
    matches = match_features(descriptors1, descriptors2)

    # If enough matches are found
    if len(matches) > 10:
        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image1 to image2 perspective
        warped_image = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))

        # Combine image2 and the warped image
        warped_image[:, 0:image2.shape[1]] = image2

        return warped_image
    else:
        return None

def stitch_frames():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    panorama = prev_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stitched_image = image_stitch(prev_gray, gray)
        if stitched_image is not None:
            panorama = stitched_image
        prev_frame = frame
        prev_gray = gray
        ret, jpeg = cv2.imencode('.jpg', panorama)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def integral_frames():
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

        # Convert the image to bytes
        _, buffer = cv2.imencode('.jpg', integral_img)

        # Convert the buffer to bytes and yield it
        integral_img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + integral_img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stitch_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/integral_feed')
def integral_feed():
    return Response(integral_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
