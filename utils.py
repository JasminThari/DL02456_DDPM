import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def plot_images_gray_scale(images, fig_size=(8,8)):
    images = images.squeeze(1)
    plt.figure(figsize=fig_size)
    # Use the 'gray' colormap for grayscale
    plt.imshow(torch.cat([i for i in images.cpu()], dim=-1), cmap='gray')
    plt.axis('off')
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    if args.dataset_path == "MNIST":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms)
    elif args.dataset_path == "CIFAR10":
        transforms = torchvision.transforms.Compose([            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64,64)),  # remove if tranning cifar with size 32x32
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root="CIFAR10", download=True, train=True, transform=transforms)
    else:
        raise AssertionError('Dataset not known!!')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader