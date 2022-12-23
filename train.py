import torch
import numpy as np
from data_loaders import IXI
from matplotlib import pyplot as plt


def train(config):
    root = config['ixi_path']
    batch_size = config['batch_size']
    ixi_loader = IXI.get_loader(root, batch_size)

    for images in ixi_loader:
        sample_image = images[0]
        plt.imshow(np.squeeze(sample_image.numpy()), cmap='gray')
        plt.show()
        break
