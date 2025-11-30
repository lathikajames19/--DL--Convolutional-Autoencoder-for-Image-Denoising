# --DL--Convolutional-Autoencoder-for-Image-Denoising
## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

Import dataset of clean images

Add synthetic noise to create noisy inputs

Normalize image pixel values

Split into train / validation sets

Build encoder with Conv + Pooling layers

Build decoder with Upsampling + Conv layers

Use MSE or BCE loss with Adam optimizer

Train model on noisy→clean mapping

Validate and visualize reconstruction results

Evaluate using PSNR / SSIM metrics





## PROGRAM
# Name : Lathika.K
# Reg Number : 212224230140

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32,kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x

# Initialize model, loss function and optimizer
model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
print("Name: Lathika.K")
print("Register Number: 212224230140")
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: Lathika.K")
    print("Register Number: 212224230140")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)


            #Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")


# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Lathika.K")
    print("Register Number: 212224230140")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```


### OUTPUT

### Model Summary

![WhatsApp Image 2025-11-30 at 9 07 12 PM](https://github.com/user-attachments/assets/589388cb-f4ec-401b-a81a-d14b2b992501)


### Training loss


![WhatsApp Image 2025-11-30 at 9 07 38 PM](https://github.com/user-attachments/assets/fc0d1982-4248-49c5-b728-b2a79a4d1191)


## Original vs Noisy Vs Reconstructed Image

![WhatsApp Image 2025-11-30 at 9 07 51 PM](https://github.com/user-attachments/assets/b5a3589d-c1ac-424a-8bde-c2e6aebd845c)


## RESULT
Therefore, To Develope a convolutional autoencoder for image denoising application executed successfully.
