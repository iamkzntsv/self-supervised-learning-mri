from models.vae_dense import VAE


channels_in, latent_dim, channels_out = 1, 2, 1
dropout_prob = 0.2

vae = VAE(channels_in, latent_dim, channels_out, dropout_prob)


