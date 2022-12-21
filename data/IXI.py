import os
import nibabel as nib
import nibabel.processing
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from preprocessing.SliceExtractor import extractor
from utils import *


class IXI(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.samples = self.make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TO DO: implement method
        pass

    def load_image(self, fname):
        """
        Load a single image from the given path
        :param fname: path to the image
        :return: a nibabel volume
        """
        # TO DO: implement skull stripping and volume slicing
        img = nib.processing.conform(nib.load(fname), (128, 128, 128)).get_fdata()
        img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
        prob = extractor.get_prob(img)
        mask = [prob > 0.5]
        print(len(mask))

        return img

    def make_dataset(self):
        """
        Create a dataset
        :return: list of ...
        """
        fnames = load_paths(self.root)[0]
        images = []
        for i in tqdm(range(len(fnames)), desc='Loading IXI Dataset'):
            img = self.load_image(fnames[i])
            images.append(img)

        return images

    def get_loader(self):
        # TO DO
        pass


