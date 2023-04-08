import os
import torch
import numpy as np
import nibabel as nib
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from processing.SliceExtractor import ext
from utils import *
import wandb
import re
import sys


class BRATS(Dataset):

    def __init__(self, root, transform=None, preprocess_data=False):
        self.root = root
        self.transform = transform

        if not preprocess_data:
            print("Loading data from a disk...")
            self.samples = load_h5('brats_data')
            self.masks = load_h5('brats_masks')
            if len(self.samples) > 0:
                print("Data loading successful. {} images collected from BRATS dataset.".format(len(self.samples)))
            else:
                print(f"Failed fetching the data from {root}. Check the data path")
                sys.exit(0)

        else:
            self.samples, self.masks = self._make_dataset()
            print('Saving...')
            save_h5(self.samples, name='brats_data.h5')
            save_h5(self.masks, name='brats_masks.h5')
            print('Saving Successful!')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index].astype(np.float32)
        mask = self.masks[index].astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

    def _make_dataset(self):
        """
        Create a dataset by loading and slicing nifti images
        :return: list of 2D axial slices
        """
        img_paths, mask_paths = self._load_paths(self.root)
        images, masks = [], []
        for i in tqdm(range(len(img_paths)), desc='Loading BRATS Dataset'):
            img = nib.load(img_paths[i])
            mask = nib.load(mask_paths[i])
            img_sliced, mask_sliced = ext.get_slices(img, mask)

            images.extend(img_sliced)
            masks.extend(mask_sliced)

        print("{} 2D slices collected from {} images.".format(len(images), len(img_paths)))

        return images, masks

    @staticmethod
    def _load_paths(root):
        fol_names = sorted(os.listdir(root))[1:]  # on Mac add [1:] to handle .DS_Store file
        img_paths, mask_paths = [], []
        for fol_name in fol_names:
            fnames = sorted(os.listdir(os.path.join(root, fol_name)))
            img_paths.append(os.path.join(root, fol_name, fnames[-1]))
            mask_paths.append(os.path.join(root, fol_name, fnames[-2]))
        return img_paths, mask_paths


def get_loader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=False):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last
                      )
