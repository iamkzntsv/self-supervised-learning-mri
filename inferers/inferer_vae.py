import torch
from preprocessing import transforms
from models.vae import VAE
from data_loaders import brats
from matplotlib import pyplot as plt
from utils import *
import sys


def run(config):
    root, load_from_disk = config['data_path'], config['load_from_disk']
    torch.manual_seed(42)

    dataset = brats.BRATS(root, transforms.get_transform(), load_from_disk=load_from_disk)
    data_loader = brats.get_loader(dataset, batch_size=1)

    model = VAE(config['latent_dim'])
    model.load_state_dict(torch.load('trained_models/vae.pt'))
    model.eval()

    for images, masks in data_loader:
        images = torch.tensor(images, dtype=torch.float32)
        reconstruction, _, _ = model(images)

        reconstruction = reconstruction[0].squeeze().detach().numpy()
        img = images[0].squeeze().detach().numpy()
        mask = masks[0].squeeze().detach().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(reconstruction, cmap='gray')
        plt.subplot(132)
        plt.imshow(img, cmap='gray')
        plt.subplot(133)
        plt.imshow(mask, cmap='gray')
        plt.show()