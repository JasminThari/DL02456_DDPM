import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from PIL import Image

# Define the transformations
transforms1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the MNIST dataset
dataset = MNIST(root="MNIST", download=True, train=True, transform=transforms1)

# Create a directory to save the images
output_dir = "real_images/MNIST_all_train_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to keep track of the number of saved images for each label
# saved_counts = {label: 0 for label in range(10)}

# Iterate through the dataset and save 10 images for each label
for i, (data, label) in enumerate(dataset):
    # if saved_counts[label] < 10:
        # Denormalize the image data
    data = (data * 0.5) + 0.5
        # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(data)
        # Construct the filename
    filename = os.path.join(output_dir, f"image_{i}.jpg")
        # Save the image as a JPG file
    image.save(filename)
        #saved_counts[label] += 1

print("Images saved successfully.")