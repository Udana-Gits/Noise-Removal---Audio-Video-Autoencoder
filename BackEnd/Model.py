# Image Denoising Autoencoder
# This script creates three denoising autoencoders for Gaussian, Speckle, and Salt & Pepper noise

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import zipfile
import requests
from tqdm import tqdm
import cv2
import glob
from torchsummary import summary

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # Display GPU details
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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


# Custom Dataset for denoising
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir="results"):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png")))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # Load noisy image
        noisy_img = Image.open(self.noisy_files[idx]).convert('RGB')
        noisy_tensor = self.transform(noisy_img)

        # Load clean image
        clean_img = Image.open(self.clean_files[idx]).convert('RGB')
        clean_tensor = self.transform(clean_img)

        return noisy_tensor, clean_tensor


# Define the Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 256x256

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x128

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Train the model
def train_model(model_name, noisy_dir, num_epochs=50, batch_size=16):
    """Train a denoising autoencoder model"""
    print(f"\nTraining model for {model_name} noise removal...")

    # Prepare dataset and dataloader
    dataset = DenoisingDataset(noisy_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function and optimizer
    model = DenoisingAutoencoder().to(device)

    # Print model summary
    print("Model Architecture:")
    summary(model, (3, 512, 512))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Keep track of losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation"):
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}")

        # Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f"models/{model_name}_denoising_epoch_{epoch + 1}.pth")

    # Save the final model
    torch.save(model.state_dict(), f"models/{model_name}_denoising_final.pth")

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Denoising Autoencoder - Loss Curves')
    plt.legend()
    plt.savefig(f"models/{model_name}_loss_curve.png")
    plt.close()

    return model


# Visualize results
def visualize_results(model, noisy_dir, num_samples=5):
    """Visualize some denoising results"""
    model.eval()

    # Get a few random images
    all_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png")))
    sample_files = random.sample(all_files, num_samples)

    plt.figure(figsize=(15, 12))

    for i, file_path in enumerate(sample_files):
        # Get corresponding clean image
        clean_file = os.path.join("results", os.path.basename(file_path))

        # Load and prepare noisy image
        noisy_img = Image.open(file_path).convert('RGB')
        noisy_tensor = transforms.ToTensor()(noisy_img).unsqueeze(0).to(device)

        # Denoise the image
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)

        # Convert tensors to images
        denoised_img = transforms.ToPILImage()(denoised_tensor.squeeze(0).cpu())
        clean_img = Image.open(clean_file).convert('RGB')

        # Display images
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(noisy_img)
        plt.title("Noisy Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(denoised_img)
        plt.title("Denoised Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(clean_img)
        plt.title("Original Image")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/denoising_comparison_{os.path.basename(noisy_dir)}.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Step 1: Check for CUDA and install if needed
    if not torch.cuda.is_available():
        print("CUDA not available. You may need to install CUDA 12.3 and cuDNN.")
        print("Visit NVIDIA website to download CUDA Toolkit 12.3: https://developer.nvidia.com/cuda-downloads")
        print("And cuDNN: https://developer.nvidia.com/cudnn")
    else:
        print("CUDA is available! Using GPU acceleration.")

    # Step 2: Download and process DIV2K dataset
    if download_div2k():
        process_div2k_images()

    # Step 3: Train models for each noise type
    # Gaussian noise model
    gaussian_model = train_model("gaussian", "gaussian_noisy")
    visualize_results(gaussian_model, "gaussian_noisy")

    # Speckle noise model
    speckle_model = train_model("speckle", "speckle_noisy")
    visualize_results(speckle_model, "speckle_noisy")

    # Salt and pepper noise model
    salt_pepper_model = train_model("salt_pepper", "salt_pepper_noisy")
    visualize_results(salt_pepper_model, "salt_pepper_noisy")

    print("Training complete! Models saved in the 'models' directory.")
    print("Evaluation results saved in the 'results' directory.")