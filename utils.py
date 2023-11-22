import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

print("Vi er gode tosser")
print("Kom nu videre, du er sååå langsom")

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):

    if args.dataset_path == "MNIST":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dataset = torchvision.datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms)


    elif args.dataset_path == "CIFAR10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root="CIFAR10", download=True, train=True, transform=transforms)

    elif args.dataset_path == "landscape_img_folder":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            # torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            # torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    else:
        raise AssertionError('Dataset not known!!')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader