import nibabel as nib
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from preprocessing.SliceExtractor import ext
from matplotlib import pyplot as plt
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

    def _extract_slices(self, fname):
        """
        Get relevant 2D slices from a 3D volume
        :param fname: path to the image
        :return:
        """
        img = load_nifti(fname)
        img_sliced = ext.extract(img)
        return img_sliced

    def _make_dataset(self):
        """
        Create a dataset
        :return: list of ...
        """
        fnames = load_paths(self.root)[0]
        images = []
        for i in tqdm(range(len(fnames)), desc='Loading IXI Dataset'):
            img_sliced = self._extract_slices(fnames[i])
            print()
            return

        return images

    def _get_loader(self):
        # TO DO
        pass


