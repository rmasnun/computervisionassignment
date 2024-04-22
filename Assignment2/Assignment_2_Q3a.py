import cv2
import numpy as np

def feature_matching(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

if __name__ == "__main__":
    # Load two images
    img1 = cv2.imread('1.jpg')  # Replace 'image1.jpg' with the path to your first image
    img2 = cv2.imread('2.jpg')  # Replace 'image2.jpg' with the path to your second image

    # Perform feature matching
    matched_img = feature_matching(img1, img2)

    # Display the matched image
    cv2.imshow('Feature Matching', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
