import torch
import torch.nn as nn
from torch.nn import functional as F
from losses.loss_vae import LossVAE
from utils import calc_activation_shape


device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):

    def __init__(self, in_channels, latent_dim, dropout_rate=0.2):
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

    def __init__(self, latent_dim):
        """
        :param latent_dim: size of the latent space
        """
        super(Decoder, self).__init__()
        self.out_channels = 1
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
                nn.Conv2d(filters[-1], self.out_channels,
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU())

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
            ResidualBlock(64, 64, 3),
        )

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                   nn.LeakyReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1),
                                   nn.LeakyReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1),
                                   nn.LeakyReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1),
                                   nn.LeakyReLU())

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        residual = self.residual_blocks(conv1)
        result = self.conv2(residual)
        result += self.conv1(inputs)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = torch.sigmoid(self.conv5(result))
        return result


class MSVAE(nn.Module):

    def __init__(self, latent_dim):
        super(MSVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(1, latent_dim)
        self.decoder = Decoder(latent_dim)
        self.super_resolution = SuperResolution().to(device)

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        super_x_hat = self.super_resolution(x_hat)
        return super_x_hat, x_hat, mu, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        return z


class LossMSVAE(nn.Module):

    def __init__(self, sigma=0.1):
        super(LossMSVAE, self).__init__()
        self.sigma = sigma

    def forward(self, super_x_hat, x_hat, x, mu, log_var):
        loss = LossVAE.kl_divergence(mu, log_var) + self.reconstruction_loss(x_hat, x) + self.l1_loss(super_x_hat, x_hat)
        return loss

    def reconstruction_loss(self, x_hat, x):
        """
        Compute the reconstruction loss
        :param x: 2D
        :param x_hat: output of the decoder, considered as the mean of a distribution
        :param scale_var: a small number for scaling variance
        """
        x = torch.reshape(x, (x.shape[0], -1)).to(device)
        x_hat = torch.reshape(x_hat, (x_hat.shape[0], -1)).to(device)
        var = torch.ones(x.size()).to(device) * self.sigma

        criterion = nn.GaussianNLLLoss(reduction='none').to(device)
        return torch.mean(torch.sum(criterion(x, x_hat, var), dim=-1))

    @staticmethod
    def l1_loss(super_X_hat, X_hat):
        """
        Objective function of the second network, aims to maximize resolution of the output image
        :param super_X_hat: output of the super resolution model
        :param X_hat: reconstruction produced by the VAE model
        """

        criterion = nn.L1Loss(reduction='sum')
        return criterion(super_X_hat, X_hat)
