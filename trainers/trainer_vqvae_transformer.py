import torch
import torch.nn as nn
import torch.optim as optim
from data_loaders import ixi
from models.vqvae import VQVAE
from models.vqvae_transformer import Transformer
from processing.transforms import get_transform
from generative.utils.enums import OrderingType
from generative.utils.ordering import Ordering
from generative.inferers import VQVAETransformerInferer
from utils import *


def make(config):
    root, preprocess_data = config['train_data_path'], config['preprocess_data']

    hyperparameters = config['hyperparameters']

    batch_size, lr = hyperparameters['optim']['batch_size'], hyperparameters['optim']['lr']

    latent_dim = 128

    transform = get_transform()
    ixi_dataset = ixi.IXI(root, transform, preprocess_data=preprocess_data)
    ixi_train_loader, ixi_valid_loader = ixi.get_loader(ixi_dataset, batch_size)

    # Instantiate the VQ-VAE
    vqvae_model = VQVAE(latent_dim=latent_dim, embedding_dim=64)
    vqvae_model.load_state_dict(torch.load(f'trained_models/vqvae_{latent_dim}.pt'))  # if CPU add param: map_location=torch.device('cpu')

    # Get the tokens ordering
    test_data = next(iter(ixi_train_loader))
    spatial_shape = vqvae_model.vqvae.encode_stage_2_inputs(test_data).shape[2:]
    ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + spatial_shape)

    # Transformer network, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure seed is the same for model initialization if multi-host training used
    seed_value = 42
    set_seed(seed_value)

    # Initialize the Transformer model
    transformer_model = Transformer(spatial_shape=spatial_shape, latent_dim=latent_dim, **hyperparameters['model'])

    inferer = VQVAETransformerInferer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=transformer_model.parameters(), lr=lr)

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

    print('Training VQ-VAE Transformer Model...')
    vqvae_model.to(device)
    transformer_model.to(device)

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
            logits, target, _ = inferer(images, vqvae_model.vqvae, transformer_model.transformer, ordering, return_latent=True)
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
                logits, quantizations_target, _ = inferer(images, vqvae_model.vqvae, transformer_model.transformer, ordering, return_latent=True)
                logits = logits.transpose(1, 2)

                # Calculate the loss
                loss = criterion(logits[:, :, :-1], quantizations_target[:, 1:])

                # Update validation loss
                valid_loss += (loss.item() * wandb.config.batch_size)

        # Get average losses
        train_loss = train_loss / len(train_loader.sampler.indices)
        valid_loss = valid_loss / len(valid_loader.sampler.indices)

        # Log statistics
        wandb.log({'train_loss': float(train_loss),
                   'val_loss': float(valid_loss),
                   'epoch': epoch})

        print('Epoch: {}, \tTraining Loss: {:.6f}, \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

    if save_model:
        torch.save(transformer_model.state_dict(),  f"vqvae_transformer_{config['latent_dim']}.pt")
