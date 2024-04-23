import cv2
import numpy as np

def load_images(image_path1, image_path2):
    # Load images from file paths
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    if image1 is None or image2 is None:
        print("Error: Unable to load images.")
        return None, None
    return image1, image2

def resize_images(image1, image2):
    # Resize images to have the same dimensions
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    resized_image1 = cv2.resize(image1, (width, height))
    resized_image2 = cv2.resize(image2, (width, height))
    return resized_image1, resized_image2

def compute_disparity(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Stereo disparity computation
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray1, gray2)

    # Normalize disparity map for display
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity_normalized

def estimate_depth(disparity_map):
    # Implement depth estimation logic here based on disparity map
    # For simplicity, let's assume a linear relationship between disparity and depth
    # You may need to calibrate your camera and adjust parameters accordingly
    # For this example, let's use a simple conversion factor
    conversion_factor = 0.1  # Example conversion factor
    depth_map = conversion_factor / (disparity_map + 1e-6)  # Adding small value to avoid division by zero
    return depth_map

def main():
    # Load images from file paths
    image_path1 = "original_image.jpeg"  # Provide the file path for the first image
    image_path2 = "moved_image.jpeg"  # Provide the file path for the second image
    image1, image2 = load_images(image_path1, image_path2)
    if image1 is None or image2 is None:
        return

    # Resize images to have the same dimensions
    image1, image2 = resize_images(image1, image2)

    # Compute stereo disparity
    disparity_map = compute_disparity(image1, image2)

    # Estimate depth from disparity map
    depth_map = estimate_depth(disparity_map)

    # Display results
    cv2.imshow("Disparity Map", disparity_map)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
