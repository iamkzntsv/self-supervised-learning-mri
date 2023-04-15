import numpy as np
import torch
from models.vqvae import VQVAE
from models.vqvae_transformer import Transformer
from generative.inferers import VQVAETransformerInferer
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from data_loaders import ixi, brats, ixi_synth
from matplotlib import pyplot as plt
from processing.transforms import get_transform
from processing.postprocessing import postprocessing_pipeline
import seaborn as sns

import sys


def run(config, data='ixi_synth'):
    torch.manual_seed(42)
    root, preprocess_data = config['test_data_path'], config['preprocess_data']

    latent_dim = config['latent_dim']

    # Initializing the VQ-VAE model
    vqvae_model = VQVAE(latent_dim=32, embedding_dim=64)
    vqvae_model.load_state_dict(torch.load(f'trained_models/vqvae_transformer/vqvae_32.pt'))  # if CPU add param: map_location=torch.device('cpu')
    vqvae_model.eval()

    spatial_shape = (8, 8)
    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

    # Initializing the Transformer model
    transformer_model = Transformer(spatial_shape=spatial_shape, latent_dim=32, attn_layers_dim=32, attn_layers_depth=4, attn_layers_heads=4, embedding_dropout_rate=0.4)
    transformer_model.load_state_dict(torch.load(f'trained_models/vqvae_transformer/vqvae_transformer_32.pt', map_location=torch.device('cpu')))

    inferer = VQVAETransformerInferer()

    transform = get_transform()

    if data == 'ixi_synth':
        ixi_dataset = ixi.IXI(root, transform, preprocess_data=preprocess_data)
        ixi_data_loader, _ = ixi.get_loader(ixi_dataset, batch_size=1)

        in_likelihoods = []
        for images in ixi_data_loader:
            log_likelihood = inferer.get_likelihood(
                inputs=images, vqvae_model=vqvae_model.vqvae, transformer_model=transformer_model.transformer, ordering=ordering
            )
            in_likelihoods.append(log_likelihood.sum(dim=(1, 2)).cpu().numpy())

        in_likelihoods = np.concatenate(in_likelihoods)

        ixi_synth_dataset = ixi_synth.IXISynth(root, transform)
        ixi_synth_data_loader = ixi_synth.get_loader(ixi_synth_dataset, batch_size=1)

        ood_likelihoods_1 = []
        for images, _ in ixi_synth_data_loader:
            log_likelihood = inferer.get_likelihood(
                inputs=images, vqvae_model=vqvae_model.vqvae, transformer_model=transformer_model.transformer, ordering=ordering
            )
            ood_likelihoods_1.append(log_likelihood.sum(dim=(1, 2)).cpu().numpy())

        ood_likelihoods_1 = np.concatenate(ood_likelihoods_1)

        sns.set_style("whitegrid", {"axes.grid": False})
        sns.kdeplot(in_likelihoods, bw_adjust=1, label="In-distribution", fill=True, cut=True)
        sns.kdeplot(ood_likelihoods_1, bw_adjust=1, label="In-distribution", fill=True, cut=True)
        plt.legend(loc="upper right")
        plt.xlabel("Log-likelihood")
        plt.savefig('log-likelihoods.png')

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)

        for images, masks in data_loader:
            images = torch.tensor(images, dtype=torch.float32)
            reconstruction, quantization_loss = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0, 0].detach().cpu().numpy()

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
