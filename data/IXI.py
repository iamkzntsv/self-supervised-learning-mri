import numpy as np
import nibabel as nib
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from preprocessing.SliceExtractor import ext
from utils import *


class IXI(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.samples = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TO DO: implement method
        pass

    def _make_dataset(self):
        """
        Create a dataset
        :return: list of ...
        """
        fnames = load_paths(self.root)[0]
        images = []
        for i in tqdm(range(len(fnames)), desc='Loading IXI Dataset'):
            img = load_nifti(fnames[i])
            img_sliced = ext.get_slices(img)
            images.extend(img_sliced)

        return images

    def _get_loader(self):
        # TO DO
        pass


