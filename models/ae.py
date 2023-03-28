import torch
import torch.nn as nn
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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

        self.layer_1 = self._make_layer(4, layers[0], stride=2)
        self.layer_2 = self._make_layer(8, layers[1], stride=2)
        self.layer_3 = self._make_layer(16, layers[2], stride=2)
        self.layer_4 = self._make_layer(32, layers[3], stride=2)

        self.fc = nn.Linear(2048, latent_dim)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=5, padding=2, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * Bottleneck.expansion

        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)

        x = self.bn_1(x)
        x = self.relu(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 2048)
        self.dconv_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dconv_2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dconv_3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dconv_4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.dconv_5 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        x_hat = self.relu(self.fc(z))
        x_hat = x_hat.reshape(-1, 128, 4, 4)
        x_hat = self.relu(self.dconv_1(x_hat))
        x_hat = self.relu(self.dconv_2(x_hat))
        x_hat = self.relu(self.dconv_3(x_hat))
        x_hat = self.relu(self.dconv_4(x_hat))
        x_hat = self.relu(self.dconv_5(x_hat))

        return torch.sigmoid(x_hat)


class AE(nn.Module):
    def __init__(self, latent_dim, layers=None):
        super(AE, self).__init__()

        if layers is None:
            layers = [1, 1, 1, 1]

        self.encoder = Encoder(latent_dim, layers)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
