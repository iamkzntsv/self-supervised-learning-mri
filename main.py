from config import get_config
from model_pipeline import *
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    parser = argparse.ArgumentParser(description='Choose the model to run')
    parser.add_argument('-n', '--model_name', type=str, help="Model name (e.g. vae_128)")
    parser.add_argument('-d', '--data', type=str, help="Dataset used. Options: 'brats', 'ixi_synth'")
    args = vars(parser.parse_args())

    config = get_config(mode='infer', **args)
    run_model_pipeline(config)


if __name__ == '__main__':
    main()
