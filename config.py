from utils import *

TRAIN_DATA_PATH = r''  # IXI train data path
TEST_DATA_PATH = r''  # Path to the test data, must be skull-stripped white matter normalized affine registered to 'fsaverage' T-1 weighted .nii/.mgz images


def get_config(mode='train', model_name='vae', latent_dim=128, data=None):
    epochs = 50

    dropout = 0.2
    sigma = 0.01
    preprocess_data = False

    if mode == 'infer':
        model_name, latent_dim = split_string(model_name)

    hyperparameters = get_hp_config(model_name)

    return {'mode': mode,
            'model_name': model_name,
            'data': data,
            'hyperparameters': hyperparameters,
            'input_dims': (1, 128, 128),
            'epochs': epochs,
            'dropout': dropout,
            'sigma': sigma,
            'latent_dim': int(latent_dim),
            'preprocess_data': preprocess_data,
            'train_data_path': TRAIN_DATA_PATH,
            'test_data_path': TEST_DATA_PATH,
            }


def get_hp_config(model_name):

    config_ae = {
        'optim': {'batch_size': 32,
                  'lr': 1e-3},
        'model': {'dropout': 0.2,
                  'use_batch_norm': True,
                  'layer_list': [1, 1, 1, 1]}
                }

    config_vae = {
        'optim': {'batch_size': 32,
                  'lr': 1e-3},
        'model': {'dropout': 0,
                  'use_batch_norm': False}
                 }

    config_res_vae = {
        'optim': {'batch_size': 16,
                  'lr': 1e-3},
        'model': {'dropout': 0.0,
                  'use_batch_norm': True,
                  'layer_list': [1, 1, 1, 1]}
                 }

    config_vqvae = {
        'optim': {'batch_size': 16,
                  'lr': 1e-4},
        'model': {'embedding_dim': 64}
    }

    config_vqvae_transformer = {
        'optim': {'batch_size': 64,
                  'lr': 1e-5},
        'model': {'attn_layers_dim': 32,
                  'attn_layers_depth': 4,
                  'attn_layers_heads': 4,
                  'embedding_dropout_rate': 0.4}
                  }

    configs = {'ae': config_ae,
               'vae': config_vae,
               'res_vae': config_res_vae,
               'vqvae': config_vqvae,
               'vqvae_transformer': config_vqvae_transformer}

    return configs[model_name]
