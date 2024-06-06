import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    resized_image = cv2.resize(binary_image, (128, 64))  # Resize to a standard size
    return resized_image

def compare_signatures(image1, image2):
    # Ensure the images are of the same size
    assert image1.shape == image2.shape, "Images must be the same size to compare"
    
    # Calculate the Structural Similarity Index (SSI) between the two images
    score, diff = ssim(image1, image2, full=True)
    return score

def verify_signature(image_path1, image_path2, threshold=0.9):
    # Preprocess the images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)
    
    # Compare the signatures
    matching_score = compare_signatures(img1, img2)
    print(f'Matching Score: {matching_score * 100:.2f}%')
    
    # Check if the matching score is above the threshold
    if matching_score >= threshold:
        print("Verified Check")
    else:
        print("Not Verified")

# Input image paths
image_path1 = 'path/to/first_signature.png'
image_path2 = 'path/to/second_signature.png'

# Verify the signatures
verify_signature(image_path1, image_path2)
