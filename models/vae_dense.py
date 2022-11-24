import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_channels, latent_dim, dropout_rate):
        """
        :param in_channels: number of channels in the input shape
        :param latent_dim: size of latent space (int)
        :param dropout_rate: dropout probability (float)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Dimensions of hidden layers
        hidden_dims = [32, 64, 128, 128]

        # Build the encoder
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=5, strides=2, padding=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=16, kernel_size=1, strides=1, padding=2),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            )
        )

        self.encoder = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            nn.LazyLinear(1024)
        )
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_sigma = nn.LazyLinear(latent_dim)

    def encode(self, input):
        # Compute encoder output
        out = self.encoder(input)
        out = torch.flatten(out, start_dim=1)
        out = self.final_layer(out)
        mu = self.fc_mu(out)

        return [mu, sigma]


    def reparameterize(self, mu, log_sigma):
        """
        :param mu:
        :param log_sigma:
        :return:
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, x):
        # Get mean and log variance
        mu, sigma = encode(x)
        # Sample
        latent_z = self.reparameterize(mu, log_sigma)

        return [latent_z, mu, sigma]


class Decoder(nn.Module):

    def __init__(self, out_channels, latent_dim):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, 1024)

        # Dimensions of hidden layers
        hidden_dims = [128, 128, 64, 32, 32]

        # Build the decoder
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=5, strides=2, padding=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1, strides=1, padding=2),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            )
        )

        self.decoder = nn.Sequential(*layers)

    def decode(self, z):
        """
        :param z: latent code
        :return:
        """
        result = self.decoder_input(z)
        result = result.view(-1, 16, 8, 8)
        result = self.decoder(result)

        return result


class VAE(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass


