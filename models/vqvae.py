import torch
from generative.networks.nets import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vqvae():
    vqvae = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 256),
        num_res_channels=256,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings=256,
        embedding_dim=32,
    )
    return vqvae



