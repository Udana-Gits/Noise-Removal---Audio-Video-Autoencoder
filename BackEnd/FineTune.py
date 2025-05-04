import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ✅ Ensure output directory exists
os.makedirs("gaussian_noisy_02", exist_ok=True)

# ✅ Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ✅ Process images in Dataset2
def process_dataset(source_dir="Dataset2"):
    files = glob.glob(os.path.join(source_dir, "*.png")) + glob.glob(os.path.join(source_dir, "*.jpg"))
    print(f"Found {len(files)} images in '{source_dir}'")

    for i, img_path in enumerate(tqdm(files)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))  # Resize to 512x512

        noisy_img = add_gaussian_noise(img)

        filename = f"{i:04d}_img.png"
        save_path = os.path.join("gaussian_noisy_02", filename)
        cv2.imwrite(save_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))

    print("✅ All images saved to 'gaussian_noisy_02'")

# ✅ Run the processing
if __name__ == "__main__":
    process_dataset()
