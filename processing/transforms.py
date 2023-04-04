import torch
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage


class Normalize:
    def __call__(self, image):
        return (image - torch.min(image)) / (torch.max(image) - torch.min(image))


def get_transform():
    return Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            ])
