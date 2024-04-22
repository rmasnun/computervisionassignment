import cv2
import numpy as np

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

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize variables for the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
panorama = prev_frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform image stitching
    stitched_image = image_stitch(prev_gray, gray)
    
    # If stitching is successful, update panorama
    if stitched_image is not None:
        panorama = stitched_image

    # Display the panorama
    cv2.imshow('Panoramic Output', panorama)

    # Update the previous frame and grayscale image for next iteration
    prev_frame = frame
    prev_gray = gray

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
