import numpy as np
import nibabel as nib
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from preprocessing.SliceExtractor import ext
from utils import *
import sys


class IXI(Dataset):

    def __init__(self, root, transform=None, preprocess_data=False):
        self.root = root
        self.transform = transform

        if not preprocess_data:
            print("Loading preprocessed data...")
            self.samples = load_h5(root)
            if len(self.samples) > 0:
                print("Data loading successful. {} images collected from IXI dataset.".format(len(self.samples)))
            else:
                print("Data loading error. Check the data path")
                sys.exit(0)

        else:
            self.samples = self._make_dataset()
            print(len(self.samples))
            print('Saving...')
            save_h5(self.samples, name='ixi_data.h5')
            print('Saving Successful!')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index].astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _make_dataset(self):
        """
        Create a dataset by loading and slicing nifti images
        :return: list of 2D axial slices
        """
        fnames = self._load_paths(self.root)[0]
        images = []
        for i in tqdm(range(len(fnames)), desc='Loading IXI Dataset'):
            img = nib.load(fnames[i])
            img_sliced = ext.get_slices(img, vol_id=i, mask=None)
            images.extend(img_sliced)

        print("{} 2D slices collected from {} images.".format(len(images), len(fnames)))
        return images

    @staticmethod
    def _load_paths(path):
        return [glob.glob(path + "/*.mgz", recursive=True)]


def get_loader(dataset, batch_size, valid_size=0.2, num_workers=1, drop_last=False, seed=50):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get indices for training and validation data
    num_train = len(dataset)
    split_pt = int(num_train * valid_size)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split_pt:], indices[:split_pt]

    # Define samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Define data loaders
    train_loader = DataLoader(dataset=dataset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=drop_last
                              )

    valid_loader = DataLoader(dataset=dataset,
                              sampler=valid_sampler,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=drop_last
                              )

    return train_loader, valid_loader
