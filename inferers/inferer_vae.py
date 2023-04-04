import numpy as np
import torch
from models.vae import VAE
from data_loaders import ixi, brats
from matplotlib import pyplot as plt
from processing.transforms import get_transform
from processing.postprocessing import postprocessing_pipeline


def run(config, data='ixi_synth'):
    root, preprocess_data = config['data_path'], config['preprocess_data']
    torch.manual_seed(42)

    transform = get_transform()

    latent_dim = 512
    model = VAE(latent_dim)
    model.load_state_dict(torch.load(f'trained_models/vae_{latent_dim}.pt'))
    model.eval()

    if data == 'ixi_synth':
        dataset = ixi.IXI(root, transform, preprocess_data=preprocess_data)
        data_loader, _ = ixi.get_loader(dataset, batch_size=1)

        for images in data_loader:
            reconstruction, _, _ = model(images)

            reconstruction = reconstruction[0].squeeze().detach().numpy()
            img = images[0].squeeze().detach().numpy()

            residual = np.abs(img - reconstruction, dtype=np.float32)
            residual[residual < 0.2] = 0
            residual[residual > 0] = 1

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(img, cmap='gray')
            plt.subplot(132)
            plt.imshow(reconstruction, cmap='gray')
            plt.subplot(133)
            plt.imshow(residual, cmap='gray')
            plt.show()

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)

        c = 0
        for images, masks in data_loader:
            reconstruction, _, _ = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0].squeeze().detach().numpy()

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, reconstruction, 30)

            c += 1

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(img, cmap='gray')
            plt.subplot(132)
            plt.imshow(residual, cmap='gray')
            plt.subplot(133)
            plt.imshow(refined_mask, cmap='gray')
            plt.show()