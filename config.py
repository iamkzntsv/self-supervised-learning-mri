from utils import *

TRAIN_DATA_PATH = r'C:\Users\sk768\Desktop\ixi_data\train'  # IXI train data path
TEST_DATA_PATH = r'C:\Users\sk768\Desktop\brats_data'  # Path to the test data, must be skull-stripped white matter normalized affine registered to 'fsaverage' T-1 weighted .nii/.mgz images


def get_config(mode='train', model_name='vae', latent_dim=128):
    epochs = 50
    dropout = 0.2
    sigma = 0.01
    preprocess_data = False

    if mode == 'train':
        sweep_configuration = get_sweep_config(model_name)
    elif mode == 'test':
        sweep_configuration = None
        model_name, latent_dim = split_string(model_name)

    return {'mode': mode,
            'model_name': model_name,
            'sweep_configuration': sweep_configuration,
            'input_dims': (1, 128, 128),
            'epochs': epochs,
            'dropout': dropout,
            'sigma': sigma,
            'latent_dim': latent_dim,
            'preprocess_data': preprocess_data,
            'train_data_path': TRAIN_DATA_PATH,
            'test_data_path': TEST_DATA_PATH,
            }


def get_sweep_config(model_name):
    config_ae = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [64]},
            'lr': {'values': [1e-3]},
            'latent_dim': {'values': [128]}
        }
    }

    config_vae = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [32, 64, 128]},
            'lr': {'values': [1e-3, 1e-4]},
            'dropout': {'values': [0.0, 0.3]},
            'use_batch_norm': {'values': [True, False]}
        }
    }

    config_res_vae = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [64]},
            'lr': {'values': [1e-3]},
            'latent_dim': {'values': [128]}
        }
    }

    config_vqvae = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'lr': {'values': [1e-3, 1e-4]},
            'embedding_dim': {'values': [16, 32, 64]}
        }
    }

    config_vqvae_transformer = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [32, 64]},
            'lr': {'values': [1e-3, 1e-4, 1e-5]},
            'latent_dim': {'values': [128]},
            'attn_layers_dim': {'values': [128, 64, 32]},
            'attn_layers_depth': {'values': [4, 8, 12]},
            'attn_layers_heads': {'values': [4, 8]},
            'embedding_dropout_rate': {'values': [0.0]}
        }
    }

    configs = {'ae': config_ae,
               'vae': config_vae,
               'res_vae': config_res_vae,
               'vqvae': config_vqvae,
               'vqvae_transformer': config_vqvae_transformer}

    return configs[model_name]
