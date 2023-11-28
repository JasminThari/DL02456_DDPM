import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from PIL import Image

# Define the transformations
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the CIFAR10 dataset
dataset_cifar = CIFAR10(root="CIFAR10", download=True, train=True, transform=transforms_cifar)

# Create a directory to save the images
output_dir_cifar = "real_images/CIFAR10_all_train_images"
os.makedirs(output_dir_cifar, exist_ok=True)

for i, (image, label) in enumerate(dataset_cifar):
    image = transforms.ToPILImage()(image)

    image_path = os.path.join(output_dir_cifar, f'image_{i}.jpg')

    image.save(image_path)


print("CIFAR10 images saved successfully.")
