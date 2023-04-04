from config import get_config
from model_pipeline import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    config = get_config(mode='train')
    model = model_pipeline(config)


if __name__ == '__main__':
    main()
