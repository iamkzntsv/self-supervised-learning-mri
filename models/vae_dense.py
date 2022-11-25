import torch
import torch.nn as nn
import torch.distributions as dist

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        filters = [32, 64, 128, 128]

        # Build the encoder
        layers = []
        for f in filters:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, f, kernel_size=5, strides=2, padding=2),
                    nn.BatchNorm2d(f),
                    nn.LeakyReLU()
                )
            )
            in_channels = f

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=1, strides=1, padding=2),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            )
        )

        self.encoder = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            nn.LazyLinear(1024)
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_sigma = nn.Linear(1024, latent_dim)

    def encode(self, input):
        """

        :param input:
        :return:
        """
        # Compute encoder output
        out = self.encoder(input)
        out = torch.flatten(out, start_dim=1)
        out = self.final_layer(out)
        mu = self.fc_mu(out)
        log_sigma = self.fc_sigma(out)

        return [mu, log_sigma]

    def reparameterize(self, mu, log_sigma):
        """
        Reparameterization trick
        :param mu:
        :param log_sigma:
        :return:
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, X):
        """
        Sample z from a distribution q
        :param X:
        :return:
        """
        # Get mean and log variance
        mu, log_sigma = self.encode(X)
        # Sample
        z = self.reparameterize(mu, log_sigma)

        return [z, dist.Normal(mu, log_sigma)]


class Decoder(nn.Module):

    def __init__(self, out_channels, latent_dim):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        # Dimensions of hidden layers
        filters = [128, 128, 64, 32, 32]

        # Build the decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        layers = []
        for i in range(len(filters) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(filters[i], filters[i + 1], kernel_size=5, strides=2, padding=2),
                    nn.BatchNorm2d(filters[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.final_layer = (
                nn.Sequential(
                    nn.Conv2d(filters[-1], out_channels, kernel_size=1, strides=1, padding=2),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """
        Reconstruct the image from the latent code
        :param z: latent code
        :return:
        """
        result = self.decoder_input(z)
        result = result.view(-1, 16, 8, 8)
        result = self.decoder(result)
        rec = self.final_layer(result)

        return rec


class VAE(nn.Module):

    def __init__(self, input_size, latent_dim, out_channels, dropout_rate):
        super().__init__()
        self.encoder = Encoder(input_size, latent_dim, dropout_rate)
        self.decoder = Decoder(out_channels, latent_dim)

    def forward(self, X):
        z, qzx = self.encoder(X)
        x_hat = self.decoder(z)
        return x_hat, z, qzx

    def encode(self, X):
        z, _ = self.encoder(X)
        return z

    def decode(self, z):
        return self.decoder(z)

