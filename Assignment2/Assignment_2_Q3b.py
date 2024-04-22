import cv2
import numpy as np

def compute_homography(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Initialize BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher()

    # Match descriptors between the two images
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography matrix using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

if __name__ == "__main__":
    # Load the two images
    img1 = cv2.imread('1.jpg')  # Replace 'image1.jpg' with the path to your first image
    img2 = cv2.imread('2.jpg')  # Replace 'image2.jpg' with the path to your second image

    # Compute the Homography matrix
    H = compute_homography(img1, img2)

    # Compute the inverse of the Homography matrix
    H_inv = np.linalg.inv(H)

    print("Homography matrix:")
    print(H)
    print("\nInverse of the Homography matrix:")
    print(H_inv)
