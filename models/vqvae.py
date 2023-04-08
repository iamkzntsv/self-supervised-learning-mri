import torch
import torch.nn as nn
from generative.networks.nets import VQVAE as VQVAEMODEL
from monai.networks.layers import Act

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQVAE(nn.Module):
    def __init__(self, latent_dim, embedding_dim=32, commitment_cost=0.01, dropout_rate=0.2):
        super(VQVAE, self).__init__()

        num_channels = (32, 64, 128, 128)
        num_res_channels = (32, 64, 128, 128)
        num_res_layers = 0
        downsample_parameters = ((2, 5, 1, 2), (2, 5, 1, 2), (2, 5, 1, 2), (2, 5, 1, 2))
        upsample_parameters = ((2, 5, 1, 2, 1), (2, 5, 1, 2, 1), (2, 5, 1, 2, 1), (2, 5, 1, 2, 1))

        self.vqvae = VQVAEMODEL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=num_channels,
            num_res_channels=num_res_channels,
            num_res_layers=num_res_layers,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            num_embeddings=latent_dim,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            dropout=dropout_rate,
            output_act=Act["sigmoid"]
        )

    def forward(self, x):
        return self.vqvae(x)


