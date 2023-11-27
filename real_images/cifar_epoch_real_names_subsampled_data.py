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
output_dir_cifar = "real_images/CIFAR10_real_images"
os.makedirs(output_dir_cifar, exist_ok=True)

# Initialize a dictionary to keep track of the number of saved images for each label
saved_counts_cifar = {label: 0 for label in range(10)}

# Iterate through the dataset and save 10 images for each label
for data, label in tqdm(dataset_cifar):
    if saved_counts_cifar[label] < 10:
        # Denormalize the image data
        data = (data * 0.5) + 0.5
        # Convert the tensor to a PIL image
        image = transforms.ToPILImage()(data)
        # Construct the filename
        filename = os.path.join(output_dir_cifar, f"label_{label}_{saved_counts_cifar[label]}.jpg")
        # Save the image as a JPG file
        image.save(filename)
        saved_counts_cifar[label] += 1

print("CIFAR10 images saved successfully.")
