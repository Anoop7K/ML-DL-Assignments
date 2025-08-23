# Autoencoders

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.ToTensor()

# Load MNIST dataset
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_loader = DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

# Autoencoder
class Autoencoder_layer(nn.Module):
    def __init__(self):
        super(Autoencoder_layer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (B,16,14,14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (B,32,7,7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (B,64,4,4)
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder_layer().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5

model.train()
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        outputs = model(images)

        # Resize outputs or images if needed to match shapes
        # In this architecture, output size is larger than input (40x40 > 28x28), so crop output
        outputs = outputs[:, :, :28, :28]

        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

# Evaluation and visualization
model.eval()
with torch.no_grad():
    dataiter = iter(data_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    outputs = outputs[:, :, :28, :28]  # Crop to input size

    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    plt.figure(figsize=(12, 4))
    for i in range(9):
        # Original images
        plt.subplot(2, 9, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')

        # Reconstructed images
        plt.subplot(2, 9, i + 10)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        plt.axis('off')

    plt.show()