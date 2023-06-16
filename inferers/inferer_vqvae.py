import os
import torch
from models.vqvae import VQVAE
from data_loaders import brats, ixi_synth
from matplotlib import pyplot as plt
from processing.transforms import get_transform
from processing.postprocessing import postprocessing_pipeline


def run(config, data='ixi_synth'):
    torch.manual_seed(42)
    root, preprocess_data = config['test_data_path'], config['preprocess_data']

    hyperparameters = config['hyperparameters']

    latent_dim = config['latent_dim']

    model = VQVAE(latent_dim=latent_dim, **hyperparameters['model'])
    model.load_state_dict(torch.load(f'trained_models/vqvae_{latent_dim}.pt', map_location=torch.device('cpu')))
    model.eval()

    transform = get_transform()

    if data == 'ixi_synth':
        dataset = ixi_synth.IXISynth(root, transform)
        data_loader = ixi_synth.get_loader(dataset, batch_size=1)

        parent_dir = 'images_processed/'
        os.mkdir(parent_dir)

        print('Performing Anomaly Detection...')
        for idx, (images, masks) in enumerate(data_loader):
            reconstruction, quantization_loss = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0, 0].detach().cpu().numpy()

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, reconstruction, 30)

            path = os.path.join('images_processed/', 'image' + str(idx))
            os.mkdir(path)

            plt.imshow(img, cmap='gray')
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'image'))

            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'residual'))

            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'detected'))

            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'gt_mask'))

        print("Anomaly Detection performed successfully. Processed images can be found in 'images_processed/'.")

    elif data == 'brats':
        dataset = brats.BRATS(root, transform, preprocess_data=preprocess_data)
        data_loader = brats.get_loader(dataset, batch_size=1)

        parent_dir = 'images_processed/'
        os.mkdir(parent_dir)

        print('Performing Anomaly Detection...')
        for idx, (images, masks) in enumerate(data_loader):
            reconstruction, quantization_loss = model(images)

            img = images[0].squeeze().detach().numpy()
            mask = masks[0].squeeze().detach().numpy()
            reconstruction = reconstruction[0, 0].detach().cpu().numpy()

            residual, binary_mask, refined_mask = postprocessing_pipeline(img, reconstruction, 30)

            path = os.path.join('images_processed/', 'image' + str(idx))
            os.mkdir(path)

            plt.imshow(img, cmap='gray')
            plt.title('Anomalous Image')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'image'))

            plt.imshow(residual, cmap='gray')
            plt.title('Residual')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'residual'))

            plt.imshow(refined_mask, cmap='gray')
            plt.title('Detected Anomaly')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'detected'))

            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'gt_mask'))

        print("Anomaly Detection performed successfully. Processed images can be found in 'images_processed/'.")
