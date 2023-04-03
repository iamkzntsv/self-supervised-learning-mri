LATENT_DIM = 128


def get_config(mode='train'):
    if mode == 'train':
        input_dims = (1, 128, 128)
        epochs = 15
        dropout = 0.2
        sigma = 0.01

        # data_path, preprocess_data = r'C:\Users\sk768\Desktop\ixi_data\train', True
        data_path, preprocess_data = r'C:\Users\sk768\Desktop\ixi_train', False

        model_name = 'vae'

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
                'lr': {'values': [1e-3]}
            }
        }

        return {'mode': mode,
                'model_name': model_name,
                'sweep_configuration': sweep_configuration,
                'input_dims': input_dims,
                'epochs': epochs,
                'dropout': dropout,
                'latent_dim': LATENT_DIM,
                'sigma': sigma,
                'data_path': data_path,
                'preprocess_data': preprocess_data,
                }

    elif mode == 'test':
        input_dims = (1, 128, 128)
        batch_size = 128

        data_path, preprocess_data = r'C:\Users\sk768\Desktop\brats_data', True
        data_path, preprocess_data = r'/Users/kuznetsov/Desktop/brats_data', True

        model_name = 'vae'

        return {'mode': mode,
                'model_name': model_name,
                'input_dims': input_dims,
                'batch_size': batch_size,
                'latent_dim': LATENT_DIM,
                'data_path': data_path,
                'preprocess_data': preprocess_data,
                }
