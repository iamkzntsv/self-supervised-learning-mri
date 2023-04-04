LATENT_DIM = 128
TRAIN_DATA_PATH = r'C:\Users\sk768\Desktop\ixi_data\train'
TEST_DATA_PATH = r'C:\Users\sk768\Desktop\brats_data'


def get_config(mode='train'):
    if mode == 'train':
        input_dims = (1, 128, 128)
        epochs = 30
        dropout = 0.2
        sigma = 0.01

        model_name = 'res_vae'
        preprocess_data = True

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
                'data_path': TRAIN_DATA_PATH,
                'preprocess_data': preprocess_data,
                }

    elif mode == 'test':
        input_dims = (1, 128, 128)
        batch_size = 128

        preprocess_data = True

        model_name = 'vae'

        return {'mode': mode,
                'model_name': model_name,
                'input_dims': input_dims,
                'batch_size': batch_size,
                'latent_dim': LATENT_DIM,
                'data_path': TEST_DATA_PATH,
                'preprocess_data': preprocess_data,
                }
