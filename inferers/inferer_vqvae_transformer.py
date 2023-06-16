import os
import numpy as np
import torch
from models.vqvae import VQVAE
from models.vqvae_transformer import Transformer
from data_loaders import brats, ixi_synth
from matplotlib import pyplot as plt
from processing.transforms import get_transform
from processing.postprocessing import postprocessing_pipeline
from generative.inferers import VQVAETransformerInferer
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
import torch.nn.functional as F


def run(config, data='ixi_synth'):
    torch.manual_seed(42)
    root, preprocess_data = config['test_data_path'], config['preprocess_data']

    latent_dim = config['latent_dim']

    # Initializing the VQ-VAE model
    vqvae_model = VQVAE(latent_dim=latent_dim, embedding_dim=64)
    vqvae_model.load_state_dict(torch.load(f'trained_models/vqvae_{latent_dim}.pt', map_location=torch.device('cpu')))
    vqvae_model.eval()

    spatial_shape = (8, 8)
    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

    # Initializing the Transformer model
    transformer_model = Transformer(spatial_shape=spatial_shape, latent_dim=latent_dim, attn_layers_dim=32, attn_layers_depth=4, attn_layers_heads=4, embedding_dropout_rate=0.4)
    transformer_model.load_state_dict(torch.load(f'trained_models/vqvae_transformer_{latent_dim}.pt'))  # if CPU add param: map_location=torch.device('cpu')

    inferer = VQVAETransformerInferer()

    transform = get_transform()

    if data == 'ixi_synth':
        dataset = ixi_synth.IXISynth(root, transform)
        data_loader = ixi_synth.get_loader(dataset, batch_size=1)

        parent_dir = 'images_processed/'
        os.mkdir(parent_dir)

        print('Performing Anomaly Detection...')
        for idx, (images, masks) in enumerate(data_loader):
            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()

            log_likelihood = inferer.get_likelihood(
                inputs=images[...].cpu(),
                vqvae_model=vqvae_model.vqvae,
                transformer_model=transformer_model.transformer,
                ordering=ordering
            )

            likelihood_mask = log_likelihood.cpu()[0, ...] < torch.quantile(log_likelihood, 0.04).item()

            mask_flattened = likelihood_mask.reshape(-1)
            mask_flattened = mask_flattened[ordering.get_sequence_ordering()]

            latent = vqvae_model.vqvae.index_quantize(images[...].cpu())
            latent = latent.reshape(latent.shape[0], -1)
            latent = latent[:, ordering.get_sequence_ordering()]
            latent = F.pad(latent, (1, 0), "constant", vqvae_model.vqvae.num_embeddings)
            latent = latent.long()
            latent_healed = latent.clone()

            for i in range(1, latent.shape[1]):
                if mask_flattened[i - 1]:
                    logits = transformer_model.transformer(latent_healed[:, :i])
                    probs = F.softmax(logits, dim=-1)
                    probs[:, :, vqvae_model.vqvae.num_embeddings] = 0
                    index = torch.argmax(probs[0, -1, :])
                    latent_healed[:, i] = index

            # reconstruct
            latent_healed = latent_healed[:, 1:]
            latent_healed = latent_healed[:, ordering.get_revert_sequence_ordering()]
            latent_healed = latent_healed.reshape((8, 8))

            image_healed = vqvae_model.vqvae.decode_samples(latent_healed[None, ...]).cpu().detach()
            image_healed = np.squeeze(image_healed[0].detach().numpy())

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, image_healed, 30)

            path = os.path.join('images_processed/', 'image' + str(idx))
            os.mkdir(path)

            plt.imshow(img, cmap='gray')
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'image'))

            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'residual'))

            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'detected'))

            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'gt_mask'))

        print("Anomaly Detection performed successfully. Processed images can be found in 'images_processed/'.")

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)

        parent_dir = 'images_processed/'
        os.mkdir(parent_dir)

        print('Performing Anomaly Detection...')
        for idx, (images, masks) in enumerate(data_loader):
            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()

            log_likelihood = inferer.get_likelihood(
                inputs=images[...].cpu(),
                vqvae_model=vqvae_model.vqvae,
                transformer_model=transformer_model.transformer,
                ordering=ordering
            )

            likelihood_mask = log_likelihood.cpu()[0, ...] < torch.quantile(log_likelihood, 0.04).item()

            # flatten the mask
            mask_flattened = likelihood_mask.reshape(-1)
            mask_flattened = mask_flattened[ordering.get_sequence_ordering()]

            latent = vqvae_model.vqvae.index_quantize(images[...].cpu())
            latent = latent.reshape(latent.shape[0], -1)
            latent = latent[:, ordering.get_sequence_ordering()]
            latent = F.pad(latent, (1, 0), "constant", vqvae_model.vqvae.num_embeddings)
            latent = latent.long()
            latent_healed = latent.clone()

            for i in range(1, latent.shape[1]):
                if mask_flattened[i - 1]:
                    logits = transformer_model.transformer(latent_healed[:, :i])
                    probs = F.softmax(logits, dim=-1)
                    probs[:, :, vqvae_model.vqvae.num_embeddings] = 0
                    index = torch.argmax(probs[0, -1, :])
                    latent_healed[:, i] = index

            latent_healed = latent_healed[:, 1:]
            latent_healed = latent_healed[:, ordering.get_revert_sequence_ordering()]
            latent_healed = latent_healed.reshape((8, 8))

            image_healed = vqvae_model.vqvae.decode_samples(latent_healed[None, ...]).cpu().detach()
            image_healed = np.squeeze(image_healed[0].detach().numpy())

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, image_healed, 30)

            path = os.path.join('images_processed/', 'image' + str(idx))
            os.mkdir(path)

            plt.imshow(img, cmap='gray')
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'image'))

            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'residual'))

            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'detected'))

            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'gt_mask'))

        print("Anomaly Detection performed successfully. Processed images can be found in 'images_processed/'.")