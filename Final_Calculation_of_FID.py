
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules_high_res import UNet
from torch.utils.tensorboard import SummaryWriter
import argparse
import wandb
from pytorch_fid import fid_score
from torchvision.models import inception_v3
from scipy.stats import entropy
from tqdm import tqdm
from datetime import datetime as dt


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



if __name__ == '__main__':
    # launch()
    device = "cuda"
    model = UNet(c_in=3, c_out=3,img_dim=32,initial_feature_maps=64,num_max_pools=2).to(device)
    ckpt = torch.load("models/CIFAR10_1_23_11/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = DiffusionProcess(img_shape=(3,32,32), img_size=32, device=device)
    x = diffusion.sampling(model, num_img= 8)

    import logging
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    
    for i in range(2048):
        logging.info(f'img: {i}')


        sampled_images = diffusion.sampling(model, num_img=1)
        folder_path_real_images = f"real_images/CIFAR10_all_train_images"
        folder_path_to_sampled_images = f"FID_Final_calculation/CIFAR10"
        if not os.path.exists(folder_path_to_sampled_images):
            os.mkdir(folder_path_to_sampled_images)

        path_to_sampled_images = os.path.join(folder_path_to_sampled_images, f"{i}.jpg")
        save_images(sampled_images, path_to_sampled_images)

        import logging
        # Initialize the logger
        logging.basicConfig(level=logging.INFO)

    # Calculate FID score
    fid_value = fid_score.calculate_fid_given_paths([folder_path_to_sampled_images, folder_path_real_images],
                                                    batch_size=1, device=device, dims=2048)

    logging.info(f'fid_value: {fid_value}')

    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()