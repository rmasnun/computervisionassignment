import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_stitch(img1, img2):
    # Convert images to Gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to select good matches
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    matches = np.asarray(good)
    
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find Homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp image1 onto image2
        warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        
        # Combine images
        stitched_image = warped_img1.copy()
        stitched_image[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        return stitched_image
    else:
        raise AssertionError("Can’t find enough keypoints for stitching.")
    

# Load all images covering 360 degrees (1,2,3)
img_paths = [f'images/image{i}.jpeg' for i in range(1, 4)]
images = [cv2.imread(img_path) for img_path in img_paths]

# Stitch images together
panorama = images[0]
for i in range(1, len(images)):
    stitched = image_stitch(panorama, images[i])
    if stitched is not None:
        panorama = stitched

# Save image
cv2.imwrite('panoramic_output1.jpg',panorama)

# Display stitched image
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()