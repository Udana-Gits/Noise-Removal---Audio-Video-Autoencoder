# Speckle Noise Denoising Autoencoder Training
# This script trains a denoising autoencoder for Speckle noise

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import glob

# Import common utilities
from model_utils import DenoisingDataset, DenoisingAutoencoder, check_gpu

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Check GPU availability
device = check_gpu()


# Train the model
def train_speckle_model(noisy_dir="speckle_noisy", num_epochs=30, batch_size=16):
    """Train a denoising autoencoder model for Speckle noise"""
    print("\nTraining model for Speckle noise removal...")

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
            }, f"models/speckle_denoising_epoch_{epoch + 1}.pth")

    # Save the final model
    torch.save(model.state_dict(), f"models/speckle_denoising_final.pth")

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Speckle Denoising Autoencoder - Loss Curves')
    plt.legend()
    plt.savefig(f"models/speckle_loss_curve.png")
    plt.close()

    return model


# Visualize results
def visualize_results(model, noisy_dir="speckle_noisy", num_samples=5):
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
    plt.savefig(f"results/denoising_comparison_speckle.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Check if the noisy images exist
    if not os.path.exists("speckle_noisy") or len(glob.glob("speckle_noisy/*.png")) == 0:
        print("Speckle noisy images not found. Please run add_noise.py first.")
    else:
        # Train the model
        speckle_model = train_speckle_model()

        # Visualize results
        visualize_results(speckle_model)

        print("Training complete! Speckle denoising model saved in the 'models' directory.")
        print("Evaluation results saved in the 'results' directory.")