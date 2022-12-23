from config import get_config
from train import train


def main(config):

    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'test':
        pass
    elif config['mode'] == 'infer':
        pass


if __name__ == '__main__':
    config = get_config()
    main(config)
