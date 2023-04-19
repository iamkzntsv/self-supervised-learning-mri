from config import get_config
from model_pipeline import *
import argparse
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # conda env create -f environment.yml
    mode = 'train'
    if mode == 'infer':
        # conda run -n myGPUenv python main.py -m test -n vae_128
        parser = argparse.ArgumentParser(description='Choose the model to run')
        parser.add_argument('-n', '--model_name', type=str, help='Model name (e.g. vae_128)')
        args = vars(parser.parse_args())

        config = get_config(mode='infer', **args)
        run_model_pipeline(config)
    else:
        model_name = 'vqvae_transformer'
        config = get_config(mode='train', model_name=model_name, latent_dim=512)
        run_model_pipeline(config)


if __name__ == '__main__':
    main()
