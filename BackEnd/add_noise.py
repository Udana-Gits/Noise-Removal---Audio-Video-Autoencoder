# Add Noise to Images
# This script processes DIV2K images and adds three types of noise: Gaussian, Speckle, and Salt & Pepper

import os
import numpy as np
import cv2
import glob
from tqdm import tqdm

# Create directories for saving noisy images
os.makedirs("gaussian_noisy", exist_ok=True)
os.makedirs("speckle_noisy", exist_ok=True)
os.makedirs("salt_pepper_noisy", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Download and extract DIV2K dataset if not already available
def download_div2k():
    print("Checking for DIV2K dataset...")
    if not os.path.exists("DIV2K"):
        # URL to download DIV2K dataset from Kaggle
        # Note: This is a placeholder. You'll need to download from Kaggle manually
        # or use the Kaggle API with proper authentication
        print("Please download DIV2K dataset from Kaggle and place it in the current directory")
        print("You can use: kaggle datasets download -d datasets/div2k")
        return False
    else:
        print("DIV2K dataset found!")
        return True


# Add different types of noise to images
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_speckle_noise(image, var=0.1):
    """Add speckle noise to image"""
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy = image + image * gauss * var
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, salt_vs_pepper=0.5, amount=0.04):
    """Add salt and pepper noise to image"""
    noisy = np.copy(image)
    # Salt (white) noise
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255

    # Pepper (black) noise
    num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy


# Process DIV2K images
def process_div2k_images(source_dir="DIV2K/DIV2K_train_HR", limit=800):
    """Process DIV2K images, resize to 512x512 and add noise"""
    files = glob.glob(os.path.join(source_dir, "*.png"))
    if not files:
        files = glob.glob(os.path.join(source_dir, "*.jpg"))

    if not files:
        print(f"No images found in {source_dir}")
        return False

    print(f"Found {len(files)} images. Processing up to {limit} images...")

    for i, img_path in enumerate(tqdm(files[:limit])):
        # Read and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (512, 512))

        # Generate filename
        filename = f"img_{i:04d}.png"

        # Save original image (for reference)
        cv2.imwrite(os.path.join("results", filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Add and save different noises
        gaussian_noisy = add_gaussian_noise(img)
        cv2.imwrite(os.path.join("gaussian_noisy", filename),
                    cv2.cvtColor(gaussian_noisy, cv2.COLOR_RGB2BGR))

        speckle_noisy = add_speckle_noise(img)
        cv2.imwrite(os.path.join("speckle_noisy", filename),
                    cv2.cvtColor(speckle_noisy, cv2.COLOR_RGB2BGR))

        salt_pepper_noisy = add_salt_pepper_noise(img)
        cv2.imwrite(os.path.join("salt_pepper_noisy", filename),
                    cv2.cvtColor(salt_pepper_noisy, cv2.COLOR_RGB2BGR))

    return True


# Main execution
if __name__ == "__main__":
    # Download and process DIV2K dataset
    if download_div2k():
        process_div2k_images()
        print("Noisy images have been generated and saved to respective folders.")
    else:
        print("Failed to process the DIV2K dataset. Please ensure it's available.")