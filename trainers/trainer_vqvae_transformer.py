import torch
import torch.nn as nn
from models.vqvae import get_vqvae
import torch.optim as optim
from data_loaders import ixi
from processing.transforms import get_transform
from models.vqvae_transformer import get_transformer_model
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from generative.inferers import VQVAETransformerInferer

import wandb
import sys


def make(config):
    root, load_from_disk = config['data_path'], config['load_from_disk']

    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    transform = get_transform()
    ixi_dataset = ixi.IXI(root, transform, load_from_disk=load_from_disk)
    ixi_train_loader, ixi_valid_loader = ixi.get_loader(ixi_dataset, batch_size)

    # Instantiate the VQ-VAE
    vqvae_model = get_vqvae()
    vqvae_model.load_state_dict(torch.load('trained_models/vqvae.pt', map_location=torch.device('cpu')))

    test_data = next(iter(ixi_train_loader))
    spatial_shape = vqvae_model.encode_stage_2_inputs(test_data).shape[2:]

    # Specifies the order in which the tokens are processed
    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

    # Transformer network, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_model = get_transformer_model(spatial_shape)
    transformer_model.to(device)

    inferer = VQVAETransformerInferer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=transformer_model.parameters(), lr=5e-4)

    return (transformer_model, vqvae_model, inferer, ordering), ixi_train_loader, ixi_valid_loader, criterion, optimizer


def train(args, train_loader, valid_loader, criterion, optimizer, config, save_model=False):
    transformer_model, vqvae_model, inferer, ordering = args

    # Log gradients and parameters of the model
    wandb.watch(transformer_model, criterion, log='all', log_freq=5)

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. \nTraining performed on CPU.')
    else:
        print('CUDA is available! \nTraining performed on GPU.')

    device = "cuda" if train_on_gpu else "cpu"

    vqvae_model.to(device)
    transformer_model.to(device)

    intermediary_images = []
    vqvae_model.eval()
    for epoch in range(config['epochs']):
        # Training loop
        transformer_model.train()
        train_loss = 0.0
        for images in train_loader:
            # Move image tensor to device
            images = images.to(device, dtype=torch.float)
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Inferer takes input images, generates its latent representation using the VQ-VAE model,
            # and then use the Transformer model to predict the next token in the latent sequence.
            logits, target, _ = inferer(images, vqvae_model, transformer_model, ordering, return_latent=True)
            logits = logits.transpose(1, 2)

            # Calculate the loss
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            # Update training_loss
            train_loss += (loss.item() * wandb.config.batch_size)

        # Validation loop
        transformer_model.eval()
        valid_loss = 0.0
        for val_step, images in enumerate(valid_loader, start=1):
            with torch.no_grad():
                # Move image tensor to device
                images = images.to(device, dtype=torch.float)
                logits, quantizations_target, _ = inferer(images, vqvae_model, transformer_model, ordering, return_latent=True)
                logits = logits.transpose(1, 2)

                # Calculate the loss
                loss = criterion(logits, target)

                # Generate a random sample to visualise progress
                if val_step == 1:
                    sample = inferer.sample(
                        vqvae_model=vqvae_model,
                        transformer_model=transformer_model,
                        ordering=ordering,
                        latent_spatial_dim=(32, 32),
                        starting_tokens=vqvae_model.num_embeddings * torch.ones((1, 1), device=device),
                    )
                    intermediary_images.append(sample[:, 0])

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
            torch.save(transformer_model.state_dict(), 'vqvae_transformer.pt')
