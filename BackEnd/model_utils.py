# Shared utilities for denoising models
# This file contains common classes and functions used by all denoising models

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os


# Custom Dataset for denoising
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir="results", img_size=512):
        # Support multiple image formats (PNG, JPG, BMP)
        self.noisy_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            self.noisy_files.extend(glob.glob(os.path.join(noisy_dir, ext)))
        self.noisy_files = sorted(self.noisy_files)
        self.clean_dir = clean_dir
        self.img_size = img_size

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # Load noisy image
        noisy_img = Image.open(self.noisy_files[idx]).convert('RGB')
        noisy_tensor = self.transform(noisy_img)

        # Get corresponding clean image using the same basename but potentially different extension
        noisy_basename = os.path.basename(self.noisy_files[idx])
        noisy_name_without_ext = os.path.splitext(noisy_basename)[0]

        # Try to find the clean image with any supported extension
        clean_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            possible_file = os.path.join(self.clean_dir, noisy_name_without_ext + ext)
            if os.path.exists(possible_file):
                clean_file = possible_file
                break

        # If we can't find a matching file, try using the same extension as the noisy file
        if clean_file is None:
            _, ext = os.path.splitext(noisy_basename)
            clean_file = os.path.join(self.clean_dir, noisy_basename)

        if not os.path.exists(clean_file):
            raise FileNotFoundError(f"Clean image not found for {noisy_basename} in {self.clean_dir}")

        clean_img = Image.open(clean_file).convert('RGB')
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


# Check GPU availability
def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Display GPU details
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available. You may need to install CUDA 12.3 and cuDNN.")
        print("Visit NVIDIA website to download CUDA Toolkit 12.3: https://developer.nvidia.com/cuda-downloads")
        print("And cuDNN: https://developer.nvidia.com/cudnn")

    return device