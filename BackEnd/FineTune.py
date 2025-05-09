import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ✅ DenoisingAutoencoder Architecture (same as your original model)
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Custom Dataset
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.noisy_images[idx])

        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img

# ✅ Parameters
noisy_dir = "gaussian_noisy_02"
clean_dir = "resized_clean_02"
batch_size = 8
epochs = 30
lr = 1e-4

# ✅ Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ✅ Dataset & DataLoader
dataset = DenoisingDataset(noisy_dir, clean_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load("models/gaussian_denoising_finetuned.pth", map_location=device))

# ✅ Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ✅ Training Loop
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for noisy_imgs, clean_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# ✅ Save the fine-tuned model
torch.save(model.state_dict(), "models/gaussian_denoising_finetuned.pth")
print("✅ Fine-tuned model saved!")
