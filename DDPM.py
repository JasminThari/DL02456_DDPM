
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

class DiffusionProcess:
    def __init__(self, img_shape = (3, 64, 64), T = 100, beta_min = 10e-4, beta_max=20e-3, device="cuda",**kwargs):
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
        :param t: timestep
        :return: noisy image, noise
        """
        noise = torch.randn(x0.shape).to(self.device)
        mu = torch.sqrt(self.alpha_bar[t]).to(self.device)
        #var = torch.full_like(mu, 1-self.alpha_bar[t])
        var = torch.full_like(mu, 1 - self.alpha_bar[t].item()).to(self.device)
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
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionProcess(**vars(args))
    model.train()
    for epoch in range(args.epochs):
        print(epoch)
        for img_batch, _ in dataloader:
            img_batch = img_batch.to(device)
            random_timestep = torch.randint(low=1, high=args.T, size=(1,)).to(device)
            noisy_image, noise = diffusion.noising(img_batch, random_timestep)
            predicted_noise = model(noisy_image, random_timestep).to(device)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        sampled_images = diffusion.sampling(model, num_img=6)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 12
    args.device = "cuda"
    args.lr = 3e-4
    # args.img_shape = (1, 28, 28)
    args.img_shape = (3,64,64)
    args.image_size = 64 # landscape fix
    args.dataset_path = "landscape_img_folder"
    args.run_name = "TheSuperiorRun"
    args.T = 1000


    train(args)

    # model = UNet().to(device)

    # Diffusion = DiffusionProcess()


    # x0 = torch.ones(5, 3, 4, 4)
    # t = 2
    # noisy_image, noise = Diffusion.noising(x0, t)

    # sampled_image = Diffusion.sampling(model,1)

    # plt.imshow(sampled_image)


