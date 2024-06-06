import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim
from io import BytesIO
from PIL import Image

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        if 'image' in response.headers['Content-Type']:
            image = Image.open(BytesIO(response.content))
            image = image.convert('L')  # Convert to grayscale
            image_np = np.array(image)
            return image_np
        else:
            with open("downloaded_content.html", "wb") as file:
                file.write(response.content)
            raise Exception(f"URL did not return an image. Content saved to 'downloaded_content.html'")
    else:
        raise Exception(f"Failed to download image from URL: {url}, status code: {response.status_code}")

def preprocess_image(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = download_image(image_path)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

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

# Input image paths (using raw content URLs)
image_path1 = 'https://raw.githubusercontent.com/chamarasab/imagerecognition/main/signature/Page0001.jpg'
image_path2 = 'https://raw.githubusercontent.com/chamarasab/imagerecognition/main/signature/Page0002.jpg'

# Verify the signatures
verify_signature(image_path1, image_path2)
