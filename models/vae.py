import torch
import torch.nn as nn
from utils import calc_activation_shape

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):

    def __init__(self, in_channels, latent_dim, dropout_rate):
        """
        :param in_channels: number of channels in the input image
        :param latent_dim: size of latent space (int)
        :param dropout_rate: dropout probability (float)
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        filters = [32, 64, 128, 128]
        ln_shape = (128, 128)

        # Build the encoder
        layers = []
        for f in filters:
            conv = nn.Conv2d(in_channels, f, kernel_size=5, stride=2, padding=2)
            ln_shape = calc_activation_shape(ln_shape, ksize=(5, 5), stride=(2, 2), padding=(2, 2))
            layer_norm = nn.LayerNorm([f, *ln_shape])
            layers.append(
                nn.Sequential(
                    conv,
                    layer_norm,
                    nn.BatchNorm2d(f),
                    nn.LeakyReLU()
                )
            )
            in_channels = f

        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=1, stride=1),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            )
        )

        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

    def encode(self, x):
        """
        Pass the input to the encoder and get the latent distribution
        :param x: data_loaders of shape (B, C, H, W)
        :return: vectors mu and log_var produced by the encoder
        """
        # Compute encoder output
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        :param mu: vector of means produced by the encoder
        :param log_var: vector of log variances produced by the encoder
        :return: sample from the distribution parametrized by mu and var
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, x):
        """
        Get the latent encoding of the data_loaders and sample z from a learned distribution
        :param x: input of shape (B, C, H, W)
        :return: sample from the distribution q_zx,
                 a list containing mu and sigma vectors
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return [z, mu, log_var]


class Decoder(nn.Module):

    def __init__(self, out_channels, latent_dim):
        """
        :param out_channels: number of channels in the reconstruction
        :param latent_dim: size of the latent space
        """
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        # Dimensions of hidden layers
        filters = [128, 128, 64, 32, 32]

        # Build the decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        layers = [nn.Sequential(
            nn.ConvTranspose2d(16, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )]

        for i in range(len(filters) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(filters[i], filters[i + 1], kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.BatchNorm2d(filters[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.final_layer = (
            nn.Sequential(
                nn.Conv2d(filters[-1], out_channels,
                          kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """
        Reconstruct the image from the latent code
        :param z: sample from the latent distribution
        :return: reconstruction of the sample z
        """
        result = self.decoder_input(z)
        result = result.reshape(-1, 16, 8, 8)
        result = self.decoder(result)
        x_hat = self.final_layer(result)

        return torch.sigmoid(x_hat)


class VAE(nn.Module):

    def __init__(self, latent_dim, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(1, latent_dim, dropout_rate)
        self.decoder = Decoder(1, latent_dim)

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        return z


class LossVAE(nn.Module):

    def __init__(self, sigma=0.1):
        super(LossVAE, self).__init__()
        self.sigma = sigma

    def forward(self, x_hat, x, mu, log_var):
        loss = self.kl_divergence(mu, log_var) + self.reconstruction_loss(x_hat, x)
        return loss

    @staticmethod
    def kl_divergence(mu, log_var):
        """
        Compute the KL divergence between given distribution q(z|x) and standard normal distribution
        :param mu: mean vector produced by the encoder, tensor of shape (B, latent_dim)
        :param log_var: log sigma vector produced by the encoder, tensor of shape (B, latent_dim)
        :return: KL divergence between q(z|x) and p(z), where p(z)~N(0,I).
        """
        kl = 0.5 * torch.sum((torch.exp(log_var) + torch.square(mu) - 1 - log_var), -1)
        return torch.mean(kl)

    def reconstruction_loss(self, x_hat, x):
        """
        Compute the reconstruction loss
        :param x: 2D
        :param x_hat: output of the decoder, considered as the mean of a distribution
        :return: reconstruction
        """
        var = torch.ones(x.size()).to(device) * self.sigma

        criterion = nn.GaussianNLLLoss(reduction='none').to(device)
        loss = torch.mean(torch.sum(criterion(x, x_hat, var).reshape(x.shape[0], -1), dim=1))
        return loss
