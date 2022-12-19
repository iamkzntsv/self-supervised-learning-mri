import os
import glob
import nibabel as nib
from deepbrain import Extractor
from torch.utils.data import Dataset, DataLoader
from utils import *


class IXI(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        # self.images = self.make_dataset(root)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass

    def load_image(self, fname):
        """
        Load a single image from the dataset
        :param fname:
        :return:
        """
        image = nib.load(fname).get_fdata()
        return image

    def make_dataset(self, root):
        """
        Create a dataset
        :param root:
        :return:
        """
        fnames = load_paths(root)
        for i in range(len(fnames[0])):
            image = self.load_image(fnames[0][i])

    def make_loader(self, ):
        pass

path = '/Users/kuznetsov/Library/CloudStorage/OneDrive-UniversityofSussex/Dissertation/Data/IXI-T1'

batch_size = 32
ixi_dataset = IXI(path)
ixi_dataset.make_dataset()
ixi_loader = DataLoader(ixi_dataset, batch_size=batch_size)