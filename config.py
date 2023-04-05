TRAIN_DATA_PATH = r'C:\Users\sk768\Desktop\ixi_data\train'  # IXI train data path
TEST_DATA_PATH = r'C:\Users\sk768\Desktop\brats_data'  # test data path


def get_config(mode='train', model_name='vae', latent_dim=128):
    epochs = 50
    dropout = 0.2
    sigma = 0.01
    preprocess_data = True

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
            'input_dims': (1, 128, 128),
            'epochs': epochs,
            'dropout': dropout,
            'latent_dim': latent_dim,
            'sigma': sigma,
            'preprocess_data': preprocess_data,
            'train_data_path': TRAIN_DATA_PATH,
            'test_data_path': TEST_DATA_PATH,
            }
