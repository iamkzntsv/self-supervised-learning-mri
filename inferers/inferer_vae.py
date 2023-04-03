import numpy as np
import torch
from preprocessing import transforms
from models.vae import VAE
from data_loaders import ixi, brats
from matplotlib import pyplot as plt
from preprocessing.transforms import get_transform
from utils import *
import sys


def run(config, data='ixi'):
    root, preprocess_data = config['data_path'], config['preprocess_data']
    torch.manual_seed(42)

    transform = get_transform()

    model = VAE(config['latent_dim'])
    # model.load_state_dict(torch.load('vae.pt'))
    model.eval()

    if data == 'ixi_synth':
        dataset = ixi.IXI(root, transform, preprocess_data=preprocess_data)
        data_loader, _ = ixi.get_loader(dataset, batch_size=1)

        for images in data_loader:
            reconstruction, _, _ = model(images)

            reconstruction = reconstruction[0].squeeze().detach().numpy()
            img = images[0].squeeze().detach().numpy()

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(img, cmap='gray')
            plt.subplot(132)
            plt.imshow(reconstruction, cmap='gray')
            plt.subplot(133)
            plt.imshow(np.abs(img - reconstruction), cmap='gray')
            plt.show()

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)
        sys.exit(1)

        for images, masks in data_loader:
            reconstruction, _, _ = model(images)

            test_image = images[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0].squeeze().detach().numpy()

            residual = normalize(np.abs(test_image - reconstruction))
            print(np.unique(residual))
            print(np.unique(residual))

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(test_image, cmap='gray')
            plt.subplot(132)
            plt.imshow(reconstruction, cmap='gray')
            plt.subplot(133)
            plt.imshow(residual)
            plt.show()


def normalize(arr):
        return (arr - np.max(arr)) / ((np.max(arr) - np.min(arr)))