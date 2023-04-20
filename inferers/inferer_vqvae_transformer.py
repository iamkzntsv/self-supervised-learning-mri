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
    vqvae_model.load_state_dict(torch.load(f'trained_models/vqvae_32.pt'))  # if CPU add param: map_location=torch.device('cpu')
    vqvae_model.eval()

    spatial_shape = (8, 8)
    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

    # Initializing the Transformer model
    transformer_model = Transformer(spatial_shape=spatial_shape, latent_dim=32, attn_layers_dim=32, attn_layers_depth=4, attn_layers_heads=4, embedding_dropout_rate=0.4)
    transformer_model.load_state_dict(torch.load(f'trained_models/vqvae_transformer_32.pt', map_location=torch.device('cpu')))

    inferer = VQVAETransformerInferer()

    transform = get_transform()
    print('cpcpcpcpcp')
    sys.exit(0)