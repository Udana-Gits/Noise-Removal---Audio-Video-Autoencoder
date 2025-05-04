import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(r"C:\Users\Hp\Desktop\KDU\Git\Noise-Removal---Audio-Video-Autoencoder\BackEnd\Real_ESRGAN-PyTorch\Real-ESRGAN")

# ✅ Step 2: Import RealESRGAN
from realesrgan.utils import RealESRGANer

from basicsr.archs.rrdbnet_arch import RRDBNet

# ✅ Step 3: Define your DenoisingAutoencoder
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

# ✅ Step 4: Set device and load your denoising model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load("models/gaussian_denoising_final.pth", map_location=device))
model.eval()

# ✅ Step 5: Load and preprocess the noisy image
image_path = "gaussian_noisy/img_0001.png"
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
noisy_tensor = transform(image).unsqueeze(0).to(device)

# ✅ Step 6: Apply denoising
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

# ✅ Step 7: Convert denoised tensor to PIL image
denoised_img = denoised_tensor.squeeze(0).cpu()
denoised_pil = transforms.ToPILImage()(denoised_img)

# ✅ Step 8: Load and run RealESRGAN for enhancement
sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)

upscaler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=sr_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

enhanced_np, _ = upscaler.enhance(np.array(denoised_pil))
enhanced_pil = Image.fromarray(enhanced_np)

# ✅ Step 9: Show all three images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy())
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Denoised Image")
plt.imshow(denoised_pil)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Enhanced with Real-ESRGAN")
plt.imshow(enhanced_pil)
plt.axis("off")

plt.tight_layout()
plt.show()
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add Real-ESRGAN repo to path
sys.path.append(
    r"C:\Users\Hp\Desktop\KDU\Git\Noise-Removal---Audio-Video-Autoencoder\BackEnd\Real_ESRGAN-PyTorch\Real-ESRGAN"
)

from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

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

# ✅ Load denoising model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load("models/gaussian_denoising_final.pth", map_location=device))
model.eval()

# ✅ Load and preprocess image
image_path = "gaussian_noisy/img_0002.png"
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
noisy_tensor = transform(image).unsqueeze(0).to(device)

# ✅ Apply denoising
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

# ✅ Convert denoised tensor to PIL
denoised_img = denoised_tensor.squeeze(0).cpu()
denoised_pil = transforms.ToPILImage()(denoised_img)

# ✅ Load and run RealESRGAN for enhancement
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

enhanced_np, _ = upscaler.enhance(np.array(denoised_pil))
enhanced_pil = Image.fromarray(enhanced_np)

# ✅ Show all images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy())
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Denoised Image")
plt.imshow(denoised_pil)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Enhanced with Real-ESRGAN")
plt.imshow(enhanced_pil)
plt.axis("off")

plt.tight_layout()
plt.show()
