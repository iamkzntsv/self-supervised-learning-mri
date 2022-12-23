import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class Loss_VAE(nn.Module):

    def __init__(self):
        super(Loss_VAE, self).__init__()

    def forward(self, X_hat, X, mu, log_var):
        loss = self.kl_divergence(mu, log_var) + self.reconstruction_loss(X_hat, X)
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

    @staticmethod
    def reconstruction_loss(X_hat, X, scale_var=0.001):
        """
        Compute the reconstruction loss
        :param X: input data_loaders
        :param X_hat: output of the decoder, considered as the mean
        :param scale_var: a small number for scaling variance
        :return: reconstruction
        """
        X = torch.reshape(X, (X.shape[0], -1))
        X_hat = torch.reshape(X_hat, (X_hat.shape[0], -1))
        var = torch.ones(X.size()).to(device) * scale_var

        criterion = nn.GaussianNLLLoss(reduction='none').to(device)
        return torch.mean(torch.sum(criterion(X, X_hat, var), dim=-1))

