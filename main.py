from config import get_config
from model_pipeline import *
import argparse
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    """
    # conda run -n myGPUenv python main.py -m test -n vae -l 128
    parser = argparse.ArgumentParser(description='Choose the model to run')
    parser.add_argument('-m', '--mode', type=str, help='Mode (train/test)')
    parser.add_argument('-n', '--model_name', type=str, help='Model name (e.g. vae_128)')
    args = parser.parse_args()

    config = get_config(mode='test')
    model = model_pipeline(config)
    """
    # wandb API key: c85c93a21cc371625da06a2c2a0b27b2061d0ba8
    # conda env create -f environment.yml
    # if train use model_name, e.g. vae, if test use model_name + latent_dim, e.g. vae_128
    mode = 'train'
    model_name = 'vqvae'

    config = get_config(mode=mode, model_name=model_name, latent_dim=128)
    model = model_pipeline(config)


if __name__ == '__main__':
    main()
