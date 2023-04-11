import numpy as np
import torch
from models.vae import VAE
from data_loaders import brats, ixi_synth
from matplotlib import pyplot as plt
from processing.transforms import get_transform
from processing.postprocessing import postprocessing_pipeline


def run(config, data='ixi_synth'):
    torch.manual_seed(42)
    root, preprocess_data = config['test_data_path'], config['preprocess_data']

    latent_dim = config['latent_dim']

    model = VAE(32)
    model.load_state_dict(torch.load(f'trained_models/vae_32.pt'))  # if CPU add param: map_location=torch.device('cpu')
    model.eval()

    transform = get_transform()

    if data == 'ixi_synth':
        dataset = ixi_synth.IXISynth(root, transform)
        data_loader = ixi_synth.get_loader(dataset, batch_size=1)

        for images, masks in data_loader:
            reconstruction, _, _ = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0].squeeze().detach().numpy()

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, reconstruction, 30)

            plt.figure(figsize=(20, 5))
            plt.subplot(141)
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            plt.subplot(142)
            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.subplot(143)
            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.subplot(144)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.show()

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)

        for images, masks in data_loader:
            reconstruction, _, _ = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0].squeeze().detach().numpy()

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, reconstruction, 30)

            plt.figure(figsize=(20, 5))
            plt.subplot(141)
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            plt.subplot(142)
            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.subplot(143)
            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.subplot(144)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.show()