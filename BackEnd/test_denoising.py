import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ✅ Add Real-ESRGAN path - using a flexible approach
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
esrgan_paths = [
    os.path.join(current_dir, 'Real_ESRGAN-PyTorch/Real-ESRGAN'),
    r"C:\Users\Hp\Desktop\KDU\Git\Noise-Removal---Audio-Video-Autoencoder\BackEnd\Real_ESRGAN-PyTorch\Real-ESRGAN"
]
for path in esrgan_paths:
    if path not in sys.path:
        sys.path.append(path)

# ✅ Patch for torchvision.transforms.functional_tensor issue
# This is a workaround for the specific import error you're encountering
import importlib.util

if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
    print("Adding compatibility patch for torchvision.transforms.functional_tensor")
    import torchvision.transforms.functional as F

    sys.modules['torchvision.transforms.functional_tensor'] = F

# ✅ Import Real-ESRGAN with error handling
upscaler = None
try:
    from realesrgan.utils import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    print("Real-ESRGAN modules imported successfully")
    has_real_esrgan = True
except ImportError as e:
    print(f"Warning: Could not import RealESRGAN modules ({e}). Make sure the path is correct.")
    has_real_esrgan = False


# ✅ Define your DenoisingAutoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ✅ Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Setup image processing
image_path = "gaussian_noisy/img_0002.png"  # You can change this to your desired image
try:
    image = Image.open(image_path).convert('RGB')
    print(f"Image loaded successfully: {image_path}")
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
noisy_tensor = transform(image).unsqueeze(0).to(device)

# ✅ Setup RealESRGAN for enhancement
if has_real_esrgan:
    try:
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

        use_half = torch.cuda.is_available()

        upscaler = RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus.pth',
            model=sr_model,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        print("RealESRGAN model loaded successfully")
    except Exception as e:
        print(f"Error setting up RealESRGAN: {e}")
        print("Continuing without enhancement...")
        upscaler = None
else:
    print("RealESRGAN not available, skipping enhancement step")

# ✅ Define all models
models = [
    {
        "name": "Gaussian Denoising",
        "path": "models/gaussian_denoising_finetuned.pth",
        "type": "autoencoder"
    },
    {
        "name": "Speckle Denoising",
        "path": "models/speckle_denoising_sharp_final.pth",
        "type": "autoencoder"
    },
    {
        "name": "Salt & Pepper Denoising",
        "path": "models/salt_pepper_denoising_sharp_final.pth",
        "type": "autoencoder"
    }
]

# Add Real-ESRGAN if available
if has_real_esrgan and upscaler:
    models.append({
        "name": "Real-ESRGAN Enhancement",
        "type": "esrgan"
    })

# ✅ Process image individually through each model
individual_results = []

print("Processing image through individual models...")
for model_info in models:
    print(f"Processing with {model_info['name']} individually...")

    if model_info["type"] == "autoencoder":
        # Load autoencoder model
        try:
            denoising_model = DenoisingAutoencoder().to(device)
            denoising_model.load_state_dict(torch.load(model_info["path"], map_location=device))
            denoising_model.eval()
            print(f"  Model loaded successfully: {model_info['path']}")
        except Exception as e:
            print(f"  Error loading model {model_info['path']}: {e}")
            continue

        # Apply denoising directly to original noisy image
        with torch.no_grad():
            result_tensor = denoising_model(noisy_tensor)

        # Convert to PIL image for visualization
        result_img = result_tensor.squeeze(0).cpu()
        result_pil = transforms.ToPILImage()(result_img)

    elif model_info["type"] == "esrgan" and upscaler:
        # Apply Real-ESRGAN enhancement directly to original noisy image
        try:
            # Convert tensor to PIL image first
            noisy_pil = transforms.ToPILImage()(noisy_tensor.squeeze(0).cpu())

            # Use RealESRGAN to enhance
            enhanced_np, _ = upscaler.enhance(np.array(noisy_pil))
            result_pil = Image.fromarray(enhanced_np)

            # Convert back to tensor for consistency
            transform_back = transforms.Compose([
                transforms.Resize((512, 512)),  # Resize back to original size
                transforms.ToTensor()
            ])
            result_tensor = transform_back(result_pil).unsqueeze(0).to(device)

        except Exception as e:
            print(f"  Error applying Real-ESRGAN enhancement: {e}")
            continue
    else:
        # This case should not happen normally
        print("  Skipping unknown model type")
        continue

    # Store individual result
    individual_results.append({
        "name": model_info["name"],
        "tensor": result_tensor,
        "image": result_pil
    })

    print(f"  Successfully applied {model_info['name']} individually")

# ✅ Process image sequentially through all models
sequential_results = []
current_tensor = noisy_tensor

# Start with just the original
sequential_results.append({
    "name": "Original Noisy Image",
    "tensor": noisy_tensor,
    "image": transforms.ToPILImage()(noisy_tensor.squeeze(0).cpu())
})

print("\nProcessing image sequentially through models...")
# First three models only (without Real-ESRGAN)
print("Processing through first three models sequentially...")
for i in range(min(3, len(models))):
    model_info = models[i]
    print(f"Adding {model_info['name']} to sequence...")

    if model_info["type"] == "autoencoder":
        # Load autoencoder model
        try:
            denoising_model = DenoisingAutoencoder().to(device)
            denoising_model.load_state_dict(torch.load(model_info["path"], map_location=device))
            denoising_model.eval()
        except Exception as e:
            print(f"  Error loading model {model_info['path']}: {e}")
            continue

        # Apply denoising to current tensor
        with torch.no_grad():
            current_tensor = denoising_model(current_tensor)

        # Convert to PIL image for visualization
        current_img = current_tensor.squeeze(0).cpu()
        current_pil = transforms.ToPILImage()(current_img)

        # Store sequential result
        sequential_results.append({
            "name": f"After Gaussian+Speckle+S&P"[:i*6+16],  # Dynamically name based on sequence
            "tensor": current_tensor.clone(),
            "image": current_pil
        })

        print(f"  Added {model_info['name']} to sequence")

# If we have Real-ESRGAN, add it as the final step
if len(models) > 3 and models[3]["type"] == "esrgan" and upscaler:
    print("Adding Real-ESRGAN as final step in sequence...")
    try:
        # Convert current tensor to PIL image first
        current_pil = transforms.ToPILImage()(current_tensor.squeeze(0).cpu())

        # Use RealESRGAN to enhance
        enhanced_np, _ = upscaler.enhance(np.array(current_pil))
        final_pil = Image.fromarray(enhanced_np)

        # Convert back to tensor for consistency
        transform_back = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize back to original size
            transforms.ToTensor()
        ])
        final_tensor = transform_back(final_pil).unsqueeze(0).to(device)

        # Store final result
        sequential_results.append({
            "name": "After All Models (G+S+SP+ESRGAN)",
            "tensor": final_tensor,
            "image": final_pil
        })

        print("  Added Real-ESRGAN to sequence")
    except Exception as e:
        print(f"  Error applying Real-ESRGAN in sequence: {e}")

# ✅ Display results
plt.figure(figsize=(20, 15))

# Calculate total number of images to display
total_images = 1 + len(individual_results) + len(sequential_results) - 1  # -1 because original is in both

# Determine optimal grid layout
if total_images <= 4:
    n_rows, n_cols = 1, total_images
elif total_images <= 8:
    n_rows, n_cols = 2, 4
else:
    n_rows, n_cols = 3, 4

# Plot index
plot_idx = 1

# Show original noisy image
plt.subplot(n_rows, n_cols, plot_idx)
plt.title("Original Noisy Image")
plt.imshow(noisy_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy())
plt.axis("off")
plot_idx += 1

# Show individual model results
for result in individual_results:
    plt.subplot(n_rows, n_cols, plot_idx)
    plt.title(f"Noisy Image through\n{result['name']}")
    plt.imshow(result["image"])
    plt.axis("off")
    plot_idx += 1

# Show sequential results (skip the original which is already shown)
for i in range(1, len(sequential_results)):
    result = sequential_results[i]
    plt.subplot(n_rows, n_cols, plot_idx)
    plt.title(f"{result['name']}")
    plt.imshow(result["image"])
    plt.axis("off")
    plot_idx += 1

plt.tight_layout()
plt.savefig("denoising_comparison_results.png", dpi=300)
plt.show()

# ✅ Save results
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}...")

# Save the original image
image_name = os.path.basename(image_path).split('.')[0]
noisy_pil = transforms.ToPILImage()(noisy_tensor.squeeze(0).cpu())
noisy_pil.save(os.path.join(output_dir, f"{image_name}_original.png"))
print(f"  Saved original image")

# Save individual results
for i, result in enumerate(individual_results):
    model_name = result["name"].lower().replace(" ", "_").replace("&", "and").replace("-", "_")
    save_path = os.path.join(output_dir, f"{image_name}_individual_{model_name}.png")
    result["image"].save(save_path)
    print(f"  Saved {save_path}")

# Save sequential results
for i, result in enumerate(sequential_results[1:], 1):  # Skip the original
    sequence_name = result["name"].lower().replace(" ", "_").replace("&", "and").replace("-", "_").replace("+", "_")
    save_path = os.path.join(output_dir, f"{image_name}_sequential_{sequence_name}.png")
    result["image"].save(save_path)
    print(f"  Saved {save_path}")

print("Denoising comparison pipeline completed!")