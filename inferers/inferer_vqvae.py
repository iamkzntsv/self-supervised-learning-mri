import numpy as np
import torch
from preprocessing import transforms
from models.vqvae import get_vqvae
from data_loaders import brats
from matplotlib import pyplot as plt
from utils import *
import sys


def run(config):
    root, load_from_disk = config['data_path'], config['load_from_disk']
    torch.manual_seed(42)

    dataset = brats.BRATS(root, transforms.get_transform(), load_from_disk=load_from_disk)
    data_loader = brats.get_loader(dataset, batch_size=1)

    model = get_vqvae()
    model.load_state_dict(torch.load('trained_models/vqvae.pt', map_location=torch.device('cpu')))
    model.eval()

    for images, masks in data_loader:
        images = torch.tensor(images, dtype=torch.float32)
        reconstruction, quantization_loss = model(images)

        img = images[0].squeeze().detach().numpy()
        mask = masks[0].squeeze().detach().numpy()

        reconstruction = reconstruction[0, 0].detach().cpu().numpy()

        residual = np.abs(reconstruction - img)

        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.imshow(img, vmin=0, vmax=1, cmap="gray")
        plt.title('Original')
        plt.subplot(142)
        plt.imshow(reconstruction, cmap='gray')
        plt.title('Reconstruction')
        plt.subplot(143)
        plt.imshow(residual, vmin=0, vmax=1, cmap="gray")
        plt.title('Residual')
        plt.subplot(144)
        plt.imshow(mask, cmap="gray")
        plt.title('Tumor Mask')
        plt.show()