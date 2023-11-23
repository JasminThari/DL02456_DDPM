
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from torch.utils.tensorboard import SummaryWriter
import argparse
import wandb
from pytorch_fid import fid_score
from torchvision.models import inception_v3
from scipy.stats import entropy

wandb.login(key="cc9eaf6580b2ef9ef475fc59ba669b2de0800b92")

class DiffusionProcess:
    def __init__(self, img_shape=(3,64,64), T=1000, beta_min=10e-4, beta_max=20e-3, device="cuda",**kwargs):
        self.img_shape = img_shape
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.T)
        self.alpha = 1-self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.device=device

    def noising(self, x0, t):
        """
        This function implements equation (4) in the paper. This will respond to forward process of the DDPM model.
        :param x0: Image without any noise
        :param t: timestep, len of t is also the batch size
        :return: noisy image, noise
        """
        noise = torch.randn(x0.shape).to(self.device)
        mu = torch.sqrt(self.alpha_bar[t]).view(x0.shape[0],1,1,1).to(self.device)
        var = (1 - self.alpha_bar[t]).view(x0.shape[0],1,1,1).to(self.device)
        noisy_image = mu * x0 + torch.sqrt(var) * noise
        return noisy_image, noise

    def sampling(self, model, num_img, variance_type=None):
        x = torch.randn((num_img, self.img_shape[0], self.img_shape[1], self.img_shape[2]),device=self.device)
        model.eval()
        with torch.no_grad():
            for iter_t in reversed(range(self.T)):
                t = (torch.ones(num_img) * iter_t).long().to(self.device)
                if iter_t > 1:
                    z = torch.randn_like(x)
                else:
                    z = 0
                    # torch.zeros_like(x)
                if variance_type is not None:
                    if variance_type == "Type2":
                        var = self.beta[iter_t]
                else:
                    var = (1 - self.alpha_bar[iter_t - 1]) / (1 - self.alpha_bar[iter_t]) * self.beta[iter_t]
                predicted_noise = model(x, t)
                x = 1 / torch.sqrt(self.alpha[iter_t]) * (x - ((1 - self.alpha[iter_t]) / (torch.sqrt(1 - self.alpha_bar[iter_t]))) * predicted_noise) + torch.sqrt(var) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet(c_in=3,c_out=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionProcess(**vars(args))
    model.train()



    for epoch in range(args.epochs):
        print(epoch)
        for img_batch, _ in dataloader:
            img_batch = img_batch.to(device)
            random_timestep = torch.randint(low=1, high=args.T, size=(img_batch.shape[0],)).to(device)
            noisy_image, noise = diffusion.noising(img_batch, random_timestep)
            predicted_noise = model(noisy_image, random_timestep).to(device)
            loss = mse(noise, predicted_noise)

            # Log training loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sampled_images = diffusion.sampling(model, num_img=6)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        wandb.log({"Training Loss": loss.item()})
        wandb.log({"Sampled Images": [wandb.Image(img) for img in sampled_images]})

        if (epoch%10)==0:

            for i in range(10):
                sampled_images_for_fid = diffusion.sampling(model, num_img=1)
                path_to_sampled_images = os.path.join("results", f"sampled_images_cifar_fid/{i}.jpg")
                save_images(sampled_images_for_fid, path_to_sampled_images)

            # Specify path to real images
            # path_to_sampled_images = os.path.join("results", f"sampled_images_fid")
            # path_to_real_images = "results/mnist_images_scaled"

            path_to_sampled_images = os.path.join("results", f"sampled_images_cifar_fid")
            path_to_real_images = "results/cifar10_images_scaled"

            # Calculate FID score
            fid_value = fid_score.calculate_fid_given_paths([path_to_sampled_images, path_to_real_images],
                                                            batch_size=1, device=device, dims=2048)

            wandb.log({"FID": fid_value, "epoch": epoch})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="CIFAR10", help="give dataset path here")
    parser.add_argument("--run_name", type=str, default="cifar10", help="give run name here to wandb")
    parser.add_argument("--batch_size", type=int, default=12, help="batchsize")
    parser.add_argument("--epochs", type=int, default=200, help="epoch size here")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--T", type=int, default=1000, help="Timestep")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    args = parser.parse_args()

    if args.dataset_path == "MNIST":
        args.img_shape = (3, 64,64)
        args.image_size = 64  # landscape fix
    elif args.dataset_path == "CIFAR10":
        args.img_shape = (3, 64, 64)
        args.image_size = 64  # landscape fix
    elif args.dataset_path == "landscape_img_folder":
        args.img_shape = (3, 64, 64)
        args.image_size = 64  # landscape fix
    else:
        raise AssertionError("UNKNOWN DATASET!!!!")

    import os
    if not os.path.exists(f"models/{args.run_name}"):
        os.mkdir(f"models/{args.run_name}")

    if not os.path.exists(f"results/{args.run_name}"):
        os.mkdir(f"results/{args.run_name}")

    wandb.init(project="DDPM_Project", name=args.run_name)#, mode="disabled")
    wandb.config.epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.lr = args.lr

    train(args)





# for filename in os.listdir(path_to_real_images):
#     image_path = os.path.join(path_to_real_images, filename)
#     image_path_final_destination = os.path.join("results/scaled_images", filename)
#     print(image_path)
#     with Image.open(image_path) as img:
#         # Resize the image
#         img = img.resize((64, 64),  Image.Resampling.LANCZOS)
#         # Save the resized image back to the folder, you can choose to overwrite or save with a different name
#         img.save(image_path_final_destination)