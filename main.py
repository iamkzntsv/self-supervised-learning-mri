from config import get_config
from train import train


def main(config):
    if config['mode'] == 'train':
        train(config)


if __name__ == '__main__':
    config = get_config()
    main(config)
