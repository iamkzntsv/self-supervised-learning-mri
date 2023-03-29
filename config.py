LATENT_DIM = 128


def get_config(mode='train'):
    if mode == 'train':
        input_dims = (1, 128, 128)
        lr = 1e-4
        epochs = 10
        batch_size = 4
        dropout = 0.2
        sigma = 0.01

        # data_path, load_from_disk = r'C:\Users\sk768\Desktop\IXI-T1', False
        data_path, load_from_disk = None, True

        model_name = 'vqvae_transformer'

        """
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {
                'goal': 'minimize',
                'name': 'val_loss'
            },
            'parameters': {
                'batch_size': {'values': [16, 64, 128]},
                'lr': {'values': [0.001, 0.005, 0.0005, 0.0001]}
            }
        }
        """

        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {
                'goal': 'minimize',
                'name': 'val_loss'
            },
            'parameters': {
                'batch_size': {'values': [64]},
                'lr': {'values': [1e-4]}
            }
        }

        return {'mode': mode,
                'model_name': model_name,
                'sweep_configuration': sweep_configuration,
                'input_dims': input_dims,
                'lr': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'dropout': dropout,
                'latent_dim': LATENT_DIM,
                'sigma': sigma,
                'data_path': data_path,
                'load_from_disk': load_from_disk,
                }

    elif mode == 'test':
        input_dims = (1, 128, 128)
        batch_size = 128

        # data_path, load_from_disk = r'C:\Users\sk768\Desktop\BraTS', False
        data_path, load_from_disk = None, True

        model_name = 'vqvae'

        return {'mode': mode,
                'model_name': model_name,
                'input_dims': input_dims,
                'batch_size': batch_size,
                'latent_dim': LATENT_DIM,
                'data_path': data_path,
                'load_from_disk': load_from_disk
                }
