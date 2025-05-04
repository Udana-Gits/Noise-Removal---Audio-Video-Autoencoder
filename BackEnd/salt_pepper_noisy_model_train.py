# Salt & Pepper Noise Denoising Autoencoder Training
# This script trains a denoising autoencoder for Salt & Pepper noise
# With improvements for preserving image sharpness

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from torchsummary import summary
import glob

# Import common utilities
from model_utils import DenoisingDataset, DenoisingAutoencoder, check_gpu


# Custom SSIM Loss
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
        # Move window to device when module is moved to device via .to(device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # Always ensure window is on the same device as input images
        window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
        self.window = window
        self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Check GPU availability
device = check_gpu()

# Define sharpening filter for post-processing
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])


def apply_sharpening(img_tensor):
    """Apply sharpening filter to tensor image"""
    # Convert tensor to numpy array
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Ensure values are in valid range
    img_np = np.clip(img_np, 0, 1)

    # Convert to uint8 for OpenCV
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Apply sharpening filter
    sharpened = cv2.filter2D(img_uint8, -1, sharpen_kernel)

    # Convert back to float and normalize
    sharpened = sharpened.astype(float) / 255.0

    # Convert back to tensor
    sharpened_tensor = torch.from_numpy(sharpened).permute(2, 0, 1).float()

    return sharpened_tensor


# Train the model
def train_salt_pepper_model(noisy_dir="salt_pepper_noisy", num_epochs=30, batch_size=16):
    """Train a denoising autoencoder model for Salt & Pepper noise with improved sharpness preservation"""
    print("\nTraining model for Salt & Pepper noise removal with sharpness preservation...")

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

    # Combined loss functions
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIM().to(device)
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

            # Combined loss (MSE + L1 + SSIM)
            mse_loss = mse_criterion(outputs, clean_imgs)
            l1_loss = l1_criterion(outputs, clean_imgs)
            ssim_loss = 1 - ssim_criterion(outputs, clean_imgs)  # Since higher SSIM is better

            # Combined loss with weights
            loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * ssim_loss

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

                # Same combined loss for validation
                mse_loss = mse_criterion(outputs, clean_imgs)
                l1_loss = l1_criterion(outputs, clean_imgs)
                ssim_loss = 1 - ssim_criterion(outputs, clean_imgs)

                # Combined loss with weights
                loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * ssim_loss

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
            }, f"models/salt_pepper_denoising_sharp_epoch_{epoch + 1}.pth")

    # Save the final model
    torch.save(model.state_dict(), f"models/salt_pepper_denoising_sharp_final.pth")

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Salt & Pepper Denoising Autoencoder - Loss Curves')
    plt.legend()
    plt.savefig(f"models/salt_pepper_sharp_loss_curve.png")
    plt.close()

    return model


# Visualize results
def visualize_results(model, noisy_dir="salt_pepper_noisy", num_samples=5):
    """Visualize some denoising results with sharpening applied"""
    model.eval()

    # Get a few random images
    all_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png")))
    sample_files = random.sample(all_files, num_samples)

    plt.figure(figsize=(20, 12))

    for i, file_path in enumerate(sample_files):
        # Get corresponding clean image
        clean_file = os.path.join("results", os.path.basename(file_path))

        # Load and prepare noisy image
        noisy_img = Image.open(file_path).convert('RGB')
        noisy_tensor = transforms.ToTensor()(noisy_img).unsqueeze(0).to(device)

        # Denoise the image
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)

        # Get the denoised image without sharpening
        denoised_tensor_cpu = denoised_tensor.squeeze(0).cpu()
        denoised_img = transforms.ToPILImage()(denoised_tensor_cpu)

        # Apply sharpening filter as post-processing
        sharpened_tensor = apply_sharpening(denoised_tensor_cpu)
        sharpened_img = transforms.ToPILImage()(sharpened_tensor)

        clean_img = Image.open(clean_file).convert('RGB')

        # Display images
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(noisy_img)
        plt.title("Noisy Image")
        plt.axis('off')

        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(denoised_img)
        plt.title("Denoised Image")
        plt.axis('off')

        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(sharpened_img)
        plt.title("Denoised + Sharpened")
        plt.axis('off')

        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(clean_img)
        plt.title("Original Image")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/denoising_comparison_salt_pepper_sharp.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Check if the noisy images exist
    if not os.path.exists("salt_pepper_noisy") or len(glob.glob("salt_pepper_noisy/*.png")) == 0:
        print("Salt & Pepper noisy images not found. Please run add_noise.py first.")
    else:
        # Train the model
        salt_pepper_model = train_salt_pepper_model()

        # Visualize results
        visualize_results(salt_pepper_model)

        print(
            "Training complete! Salt & Pepper denoising model with sharpness preservation saved in the 'models' directory.")
        print("Evaluation results saved in the 'results' directory.")