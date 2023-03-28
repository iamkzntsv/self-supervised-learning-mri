import numpy as np
import torch
import scipy.ndimage as ndi
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose
import torchvision.transforms.functional as TF


class Normalize:
    def __call__(self, image):
        return (image - torch.min(image)) / (torch.max(image) - torch.min(image))


def get_transform():
    return Compose([
            ToTensor(),
            Resize((128, 128)),
            Normalize(),
            ])
