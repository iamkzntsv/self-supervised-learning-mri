import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist


class Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 dropout_rate):

        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Define dimensions of hidden layers
        hidden_dims = [32, 64, 128, 128]

        # Encoder
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=5, strides=2, padding=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        # Bottleneck
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=16, kernel_size=1, strides=1, padding=2),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),

                nn.LazyLinear(1024)
            )
        )

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_sigma = nn.LazyLinear(latent_dim)

    def encode(self, input):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        log_sigma = self.fc_sigma(out)

        return [mu, sigma]


    def reparametrize(self):
        pass


class Decoder(nn.Module):

    def __init__(self):
        pass
