import torch
from torch import optim
from data_loaders import ixi
from models.vae import VAE, LossVAE
from processing.transforms import get_transform
from utils import *


import wandb


def make(config):
    root, preprocess_data = config['train_data_path'], config['preprocess_data']

    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    dropout = wandb.config.dropout
    use_batch_norm = wandb.config.use_batch_norm

    latent_dim = config['latent_dim']

    transform = get_transform()
    ixi_dataset = ixi.IXI(root, transform, preprocess_data=preprocess_data)
    ixi_train_loader, ixi_valid_loader = ixi.get_loader(ixi_dataset, batch_size)

    # Ensure seed is the same for model initialization if multi-host training used
    seed_value = 42
    set_seed(seed_value)

    # Instantiate the model
    model = VAE(latent_dim, dropout_rate=dropout, use_batch_norm=use_batch_norm)
    model.train()

    # Loss function and Optimizer
    criterion = LossVAE(sigma=config['sigma'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, ixi_train_loader, ixi_valid_loader, criterion, optimizer


def train(model, train_loader, valid_loader, criterion, optimizer, config, save_model=False):
    # Log gradients and parameters of the model
    wandb.watch(model, criterion, log='all', log_freq=5)

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. \nTraining performed on CPU.')
    else:
        print('CUDA is available! \nTraining performed on GPU.')

    device = "cuda" if train_on_gpu else "cpu"

    print('Training VAE Model...')
    model.to(device)
    for epoch in range(config['epochs']):

        # Training loop
        model.train()
        train_loss = 0.0
        for images in train_loader:
            # Move image tensor to device
            images = images.to(device, dtype=torch.float)
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            x_hat, mu, log_var = model(images)
            # Calculate the loss
            loss = criterion(x_hat, images, mu, log_var)
            loss.backward()
            # Update parameters
            optimizer.step()
            # Update training_loss
            train_loss += (loss.item() * wandb.config.batch_size)

        # Validation loop
        model.eval()
        valid_loss = 0.0
        for images in valid_loader:
            # Move image tensor to device
            images = images.to(device, dtype=torch.float)
            # Forward pass
            x_hat, mu, log_var = model(images)
            # Calculate the loss
            loss = criterion(x_hat, images, mu, log_var)
            # Update validation loss
            valid_loss += (loss.item() * wandb.config.batch_size)

        # Get average losses
        train_loss = train_loss / len(train_loader.sampler.indices)
        valid_loss = valid_loss / len(valid_loader.sampler.indices)

        # Print statistics
        wandb.log({'train_loss': float(train_loss),
                   'val_loss': float(valid_loss),
                   'epoch': epoch})

        print('Epoch: {}, \tTraining Loss: {:.6f}, \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

    if save_model:
        torch.save(model.state_dict(), f"vae_{config['latent_dim']}.pt")

    """
    # Sample from the learned distribution and save the result
        z = model.sample(1)
        x_hat = model.decode(z)

        plt.imshow(np.squeeze(x_hat.cpu().detach().numpy()), cmap='gray')
        plt.title('Latent dim: {}, Epoch: {}'.format(config.latent_dim, epoch + 1))
        plt.savefig('images/img_' + str(epoch + 1) + '.png')
    """
