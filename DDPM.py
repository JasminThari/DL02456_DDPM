
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

class DiffusionProcess:
    def __init__(self, img_shape = (3, 64, 64), T = 10, beta_min = 10e-4, beta_max=20e-3,device="cuda"):
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
        noise = torch.randn(x0.shape)
        mu = torch.sqrt(self.alpha_bar[t])
        var = torch.full_like(mu, 1-self.alpha_bar[t])
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

                predicted_mean_noise = model(x, t)
                x = 1 / torch.sqrt(self.alpha[iter_t]) * (x - ((1 - self.alpha[iter_t]) / (torch.sqrt(1 - self.alpha_bar[iter_t]))) * predicted_mean_noise) + torch.sqrt(var) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


# def train(self, )


if __name__ == '__main__':

    device="cuda"

    model = UNet().to(device)

    Diffusion = DiffusionProcess()


    # x0 = torch.ones(5, 3, 4, 4)
    # t = 2
    # noisy_image, noise = Diffusion.noising(x0, t)

    sampled_image = Diffusion.sampling(model,1)

    # plt.imshow(sampled_image)



