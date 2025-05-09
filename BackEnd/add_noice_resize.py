import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ✅ Create output directories
clean_dir = "resized_clean_02"
noisy_dir = "speckle_noisy_02"
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(noisy_dir, exist_ok=True)

# ✅ Speckle noise function
def add_speckle_noise(image):
    noise = np.random.randn(*image.shape) * 0.2  # Adjust noise strength as needed
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# ✅ Process and rename images
def process_dataset(source_dir="Dataset2"):
    files = sorted(glob.glob(os.path.join(source_dir, "*.png")) +
                   glob.glob(os.path.join(source_dir, "*.jpg")) +
                   glob.glob(os.path.join(source_dir, "*.bmp")))
    print(f"Found {len(files)} images in '{source_dir}'")

    for i, img_path in enumerate(tqdm(files)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        filename = f"{i:04d}_image.png"

        # ✅ Save clean resized image
        clean_path = os.path.join(clean_dir, filename)
        cv2.imwrite(clean_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # ✅ Add and save speckle noisy image
        noisy_img = add_speckle_noise(img)
        noisy_path = os.path.join(noisy_dir, filename)
        cv2.imwrite(noisy_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved {len(files)} clean images to '{clean_dir}'")
    print(f"✅ Saved {len(files)} noisy images to '{noisy_dir}'")

# ✅ Run the script
if __name__ == "__main__":
    process_dataset()
