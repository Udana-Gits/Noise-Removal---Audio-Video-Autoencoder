from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import os
import sys
from flask_cors import CORS
import uuid
import tempfile

# Add Real-ESRGAN path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Real_ESRGAN-PyTorch/Real-ESRGAN'))

# Import RealESRGAN components
try:
    from realesrgan.utils import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("Warning: Could not import RealESRGAN modules. Make sure the path is correct.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define temporary directory for saving images
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'denoised_images')
os.makedirs(TEMP_DIR, exist_ok=True)


# Define the denoising autoencoder model architecture
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


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model_paths = {
    "gaussian": os.path.join(os.path.dirname(__file__), "models/gaussian_denoising_finetuned.pth"),
    "salt_pepper": os.path.join(os.path.dirname(__file__), "models/salt_pepper_denoising_sharp_final.pth"),
    "speckle": os.path.join(os.path.dirname(__file__), "models/speckle_denoising_sharp_final.pth")
}

models = {}
for model_type, path in model_paths.items():
    try:
        model = DenoisingAutoencoder().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[model_type] = model
        print(f"Successfully loaded {model_type} model from {path}")
    except Exception as e:
        print(f"Failed to load {model_type} model: {e}")

# Initialize RealESRGAN model
try:
    sr_model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    use_half = torch.cuda.is_available()
    real_esrgan = RealESRGANer(
        scale=4,
        model_path=os.path.join(os.path.dirname(__file__), 'weights/RealESRGAN_x4plus.pth'),
        model=sr_model,
        tile=128,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device
    )
    print("RealESRGAN model loaded successfully")
except Exception as e:
    print(f"Failed to load RealESRGAN model: {e}")
    real_esrgan = None


# Function to denoise image using model
def denoise_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Denoise
    with torch.no_grad():
        output = model(img_tensor)

    # Convert tensor back to PIL image
    output_img = output.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_img)

    return output_img


# Function to enhance image using RealESRGAN
def enhance_with_real_esrgan(image):
    if real_esrgan is None:
        return image

    # Convert PIL image to numpy array
    img_np = np.array(image)

    # Process with RealESRGAN
    try:
        enhanced_np, _ = real_esrgan.enhance(img_np)
        enhanced_pil = Image.fromarray(enhanced_np)

        # Resize back to original size if needed
        enhanced_pil = enhanced_pil.resize((512, 512), Image.LANCZOS)
        return enhanced_pil
    except Exception as e:
        print(f"Error in RealESRGAN processing: {e}")
        return image


# Function to convert PIL image to base64
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# Save image and return path
def save_image(img, prefix):
    filename = f"{prefix}_{uuid.uuid4()}.png"
    filepath = os.path.join(TEMP_DIR, filename)
    img.save(filepath)
    return filepath


@app.route('/denoise', methods=['POST'])
def denoise():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Open and process the image
        original_image = Image.open(file.stream).convert('RGB')

        # Resize for processing
        resized_image = original_image.resize((512, 512), Image.LANCZOS)

        # Save original image
        original_path = save_image(resized_image, "original")

        results = {
            "original": {
                "path": original_path,
                "base64": pil_to_base64(resized_image)
            }
        }

        # Process with each model
        for model_type, model in models.items():
            denoised_image = denoise_image(resized_image, model)
            saved_path = save_image(denoised_image, model_type)
            results[model_type] = {
                "path": saved_path,
                "base64": pil_to_base64(denoised_image)
            }

        # Process with RealESRGAN
        if real_esrgan is not None:
            enhanced_image = enhance_with_real_esrgan(resized_image)
            saved_path = save_image(enhanced_image, "realesrgan")
            results["realesrgan"] = {
                "path": saved_path,
                "base64": pil_to_base64(enhanced_image)
            }

        return jsonify(results)

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<model_type>/<filename>', methods=['GET'])
def download_image(model_type, filename):
    try:
        file_path = os.path.join(TEMP_DIR, f"{model_type}_{filename}.png")
        return send_file(file_path, as_attachment=True, download_name=f"{model_type}_denoised.png")
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)