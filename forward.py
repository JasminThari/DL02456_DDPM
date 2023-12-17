#%%
import torch
from PIL import Image
import numpy as np
import os 
from utils import *

def noising(input_image_path, output_image_path, t, T):
    """
    Add noise to an input image and save the noisy image.
    
    :param input_image_path: Path to the input JPG image.
    :param output_image_path: Path to save the noisy JPG image.
    :param alpha_bar: Precomputed alpha_bar values for different timesteps.
    :param t: Timestep for adding noise.
    """
    # Load the input image
    image = Image.open(input_image_path)
    image = np.array(image)  # Convert to NumPy array

    image = torch.from_numpy(image)
    
    # If you want to normalize the pixel values to the range [0, 1], you can divide by 255.
    image = image.float() / 255.0

    beta_min = 10e-4
    beta_max = 20e-3
    beta = torch.linspace(beta_min, beta_max, T)
    alpha = 1-beta
    alpha_bar = np.cumprod(alpha, 0)
    # Add noise
    noise = np.random.randn(*image.shape)
    mu = torch.sqrt(alpha_bar[t])
    var = 1 - alpha_bar[t]
    noisy_image = mu * image + torch.sqrt(var) * noise

    # Clip values to ensure they are in the valid range [0, 255]
    #noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    noisy_image = (noisy_image.clamp(-1, 1) + 1) / 2
    noisy_image = (noisy_image * 255).type(torch.uint8)
    return noisy_image


# Define the paths to your input and output images
input_image_path = 'real_images/real_images/CIFAR10_real_images_64/label_7_4.jpg'
output_image_path = 'noisy_image_7_4.jpg'

# Example alpha_bar values for different timesteps (adjust as needed)


# Choose a timestep 't' based on your requirements
t = 500

# Add noise and save the noisy image
noisy_image = noising(input_image_path, output_image_path, t, 1000)

# %%
import matplotlib.pyplot as plt
# Convert the noisy_image tensor to a NumPy array
noisy_image_np = noisy_image.cpu().numpy()



# Ensure the values are in the [0, 1] range (if needed)
noisy_image_np = (noisy_image_np - noisy_image_np.min()) / (noisy_image_np.max() - noisy_image_np.min())

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(noisy_image_np)
plt.axis('off')  # Turn off axis labels
plt.show()