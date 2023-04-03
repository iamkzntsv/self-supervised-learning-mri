import torch
import torch.nn as nn
from models.vqvae import get_vqvae
import torch.optim as optim
from data_loaders import ixi
from preprocessing.transforms import get_transform

import wandb


def make(config):
    root, load_from_disk = config['data_path'], config['load_from_disk']

    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    transform = get_transform()
    ixi_dataset = ixi.IXI(root, transform, load_from_disk=load_from_disk)
    ixi_train_loader, ixi_valid_loader = ixi.get_loader(ixi_dataset, batch_size)

    # Instantiate the model
    model = get_vqvae()
    model.train()

    # Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
            reconstruction, quantization_loss = model(images=images)
            # Calculate the loss
            recons_loss = criterion(reconstruction.float(), images.float())
            loss = recons_loss + quantization_loss
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
            reconstruction, quantization_loss = model(images=images)
            # Calculate the loss
            recons_loss = criterion(reconstruction.float(), images.float())
            loss = recons_loss + quantization_loss
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
        torch.save(model.state_dict(), 'vqvae.pt')
