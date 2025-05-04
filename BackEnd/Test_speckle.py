import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add Real-ESRGAN path
sys.path.append(
    r"C:\Users\Hp\Desktop\KDU\Git\Noise-Removal---Audio-Video-Autoencoder\BackEnd\Real_ESRGAN-PyTorch\Real-ESRGAN"
)

from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ✅ Step 1: Load the noisy image
image_path = "gaussian_noisy/img_0002.png"
image = Image.open(image_path).convert('RGB')
original_np = np.array(image)

# ✅ Step 2: Load Real-ESRGAN model
sr_model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32, scale=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ✅ Step 3: Enhance using Real-ESRGAN
enhanced_np, _ = upscaler.enhance(original_np)
enhanced_pil = Image.fromarray(enhanced_np)

# ✅ Step 4: Show the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Noisy Image")
plt.imshow(original_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Enhanced with Real-ESRGAN")
plt.imshow(enhanced_pil)
plt.axis("off")

plt.tight_layout()
plt.show()
