# speckle_noisy_model_train.py

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

from model_utils import DenoisingDataset, DenoisingAutoencoder, check_gpu

# Custom SSIM Loss (same as before)
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

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

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
        self.window = window
        self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

def apply_sharpening(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    sharpened = cv2.filter2D(img_uint8, -1, sharpen_kernel)
    sharpened = sharpened.astype(float) / 255.0
    sharpened_tensor = torch.from_numpy(sharpened).permute(2, 0, 1).float()
    return sharpened_tensor

# Main training function
def train_speckle_model(noisy_dir="speckle_noisy", num_epochs=50, batch_size=16):
    print("\nTraining model for Speckle noise removal with sharpness preservation...")
    device = check_gpu()

    dataset = DenoisingDataset(noisy_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DenoisingAutoencoder().to(device)
    summary(model, (3, 512, 512))

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noisy_imgs)
            mse_loss = mse_criterion(outputs, clean_imgs)
            l1_loss = l1_criterion(outputs, clean_imgs)
            ssim_loss = 1 - ssim_criterion(outputs, clean_imgs)

            loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * ssim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation"):
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                outputs = model(noisy_imgs)

                mse_loss = mse_criterion(outputs, clean_imgs)
                l1_loss = l1_criterion(outputs, clean_imgs)
                ssim_loss = 1 - ssim_criterion(outputs, clean_imgs)
                loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * ssim_loss

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f"models/speckle_denoising_sharp_epoch_{epoch + 1}.pth")

    torch.save(model.state_dict(), f"models/speckle_denoising_sharp_final.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Speckle Denoising Autoencoder - Loss Curves')
    plt.legend()
    plt.savefig(f"models/speckle_sharp_loss_curve.png")
    plt.close()

    return model

# Optional visualization function could be reused from salt_pepper script

if __name__ == "__main__":
    if not os.path.exists("speckle_noisy") or len(glob.glob("speckle_noisy/*.png")) == 0:
        print("Speckle noisy images not found. Please run add_noise.py first.")
    else:
        trained_model = train_speckle_model()
        print("Training complete! Model saved.")
