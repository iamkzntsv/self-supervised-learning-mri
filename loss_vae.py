import torch
import torch.distributions as dist
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Loss_VAE:

    def __init__(self, VAE, X, latent_dim):
        self.latent_dim = latent_dim
        vae = VAE()
        X_hat, z, qzx = vae(X)
        self.elbo = self.reconstruction_loss(X, X_hat) - self.kl_divergence(qzx, z, X)

    def compute(self):
        return -self.elbo

    def kl_divergence(self, qzx, z, X):
        """
        Compute the KL divergence given batches of samples from latent code z
        :param qzx: distribution of z|x produced by the encoder
        :param z: sample from a distribution q
        :param X: batch of data instances, shape (B, C, H, W)
        :return: KL divergence between q(z|x) and p(z)
        """
        # Define prior
        pz = dist.Normal(torch.zeros((X.shape[0], self.latent_dim)).to(device), torch.ones((X.shape[0], self.latent_dim)).to(device))

        # Compute log probabilities
        log_qzx = qzx.log_prob(z)
        log_pz = pz.log_prob(z)

        # Compute KL divergence
        kl = (log_qzx - log_pz)

        return kl.sum(-1)

    def reconstruction_loss(self, X, X_hat):
        """
        Compute the reconstruction loss
        :param X:
        :param X_hat:
        :return:
        """
        return F.mse_loss(X, X_hat)

