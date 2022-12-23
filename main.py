from config import get_config
from train import train


def main():

    config = get_config()
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'infer':
        pass


if __name__ == '__main__':
    main()
