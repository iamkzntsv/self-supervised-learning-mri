import torch
import torch.nn as nn
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class EncoderBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.ln1 = nn.LayerNorm([planes, inplanes // planes, inplanes // planes]) # Replace BatchNorm2d with LayerNorm
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,
                               padding=2, bias=False)
        self.ln2 = nn.LayerNorm([planes, inplanes // (planes * stride), inplanes // (planes * stride)]) # Replace BatchNorm2d with LayerNorm
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.ln3 = nn.LayerNorm([planes * 4, inplanes // (planes * stride), inplanes // (planes * stride)]) # Replace BatchNorm2d with LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.ln1(out) # Use LayerNorm instead of BatchNorm
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out) # Use LayerNorm instead of BatchNorm
        out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out) # Use LayerNorm instead of BatchNorm

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Encoder(nn.Module):

    def __init__(self, latent_dim, layers):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.inplanes = 8

        self.bn_1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        self.layer_1 = self._make_layer(8, layers[0], stride=2)
        self.layer_2 = self._make_layer(8, layers[1], stride=2)
        self.layer_3 = self._make_layer(16, layers[2], stride=2)
        self.layer_4 = self._make_layer(32, layers[3])

        self.conv_1x1 = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True))

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * EncoderBottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * EncoderBottleneckBlock.expansion,
                          kernel_size=5, padding=2, stride=stride, bias=False),
                nn.BatchNorm2d(planes * EncoderBottleneckBlock.expansion),
            )

        layers = []
        layers.append(EncoderBottleneckBlock(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * EncoderBottleneckBlock.expansion

        for i in range(1, blocks):
            layers.append(EncoderBottleneckBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        """
        Pass the input to the encoder and get the latent distribution
        :param x: data_loaders of shape (B, C, H, W)
        :return: vectors mu and log_var produced by the encoder
        """
        # Compute encoder output
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.conv_1x1(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

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


class DecoderBottleneckBlock(nn.Module):
    expansion = 4  # expansion factor

    def __init__(self, in_channels, planes, upsample=None, stride=2, output_padding=0):
        super(DecoderBottleneckBlock, self).__init__()

        self.upsample = upsample
        self.stride = stride

        self.conv_1 = nn.ConvTranspose2d(in_channels, planes, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(planes)

        self.conv_2 = nn.ConvTranspose2d(planes, planes, kernel_size=5, stride=self.stride, padding=2,
                                         output_padding=output_padding)
        self.bn_2 = nn.BatchNorm2d(planes)

        self.conv_3 = nn.ConvTranspose2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity
        x = self.relu(x)

        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim, layer_list=[1, 1, 1, 1]):
        """
        :param latent_dim: size of the latent space
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = 16

        self.dense_1 = nn.Linear(latent_dim, 1024)

        # Build the residual decoder
        self.layer_1 = self._make_layer(layer_list[3], planes=32)
        self.layer_2 = self._make_layer(layer_list[2], planes=32, stride=2, output_padding=1)
        self.layer_3 = self._make_layer(layer_list[1], planes=16, stride=2, output_padding=1)
        self.layer_4 = self._make_layer(layer_list[0], planes=8, stride=2, output_padding=1)
        self.layer_5 = self._make_layer(1, planes=8, stride=2, output_padding=1)

        self.upconv_1 = nn.ConvTranspose2d(32, 1, kernel_size=1)

    def _make_layer(self, stack, planes, stride=1, output_padding=0):
        sub_layers = []
        upsample = None

        # Initialize upsampling
        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, planes * DecoderBottleneckBlock.expansion, kernel_size=1,
                               stride=stride, output_padding=output_padding),
            nn.BatchNorm2d(planes * DecoderBottleneckBlock.expansion)
        )

        # First stack layer
        sub_layers.append(DecoderBottleneckBlock(self.in_channels, planes, upsample=upsample, stride=stride,
                                                 output_padding=output_padding))
        self.in_channels = planes * DecoderBottleneckBlock.expansion

        # Other stack layers
        for i in range(stack - 1):
            sub_layers.append(DecoderBottleneckBlock(self.in_channels, planes, upsample=None, stride=1))

        return nn.Sequential(*sub_layers)

    def forward(self, z):
        """
        Reconstruct the image from the latent code
        :param z: sample from the latent distribution
        :return: reconstruction of the sample z
        """
        x = self.dense_1(z)
        x = x.reshape(-1, 16, 8, 8)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x_hat = self.upconv_1(x)

        return torch.sigmoid(x_hat)


class ResVAE(nn.Module):

    def __init__(self, latent_dim, layer_list=[3, 4, 6, 3]):
        super(ResVAE, self).__init__()
        self.encoder = Encoder(latent_dim, layer_list)
        self.decoder = Decoder(latent_dim, layer_list)
        self.latent_dim = latent_dim

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