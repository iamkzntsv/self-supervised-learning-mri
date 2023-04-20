from config import get_config
from model_pipeline import *
import argparse
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(mode='infer'):
    # conda env create -f environment.yml
    if mode == 'infer':
        """
        # conda run -n myGPUenv python main.py -n vqvae_transformer_32 -d brats
        parser = argparse.ArgumentParser(description='Choose the model to run')
        parser.add_argument('-n', '--model_name', type=str, help="Model name (e.g. vae_128)")
        parser.add_argument('-d', '--data', type=str, help="Dataset used. Options: 'brats', 'ixi_synth'")
        args = vars(parser.parse_args())

        config = get_config(mode='infer', **args)
        run_model_pipeline(config)
        """
        model_name = 'vqvae_transformer_64'
        config = get_config(mode='infer', model_name=model_name, data='ixi_synth')
        run_model_pipeline(config)
    else:
        model_name = 'vae'
        config = get_config(mode='train', model_name=model_name, latent_dim=8)
        run_model_pipeline(config)


if __name__ == '__main__':
    main(mode='train')
