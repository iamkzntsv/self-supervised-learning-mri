import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import os


class IXISynth(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        print("Loading data from a disk...")
        self.samples = load_h5(os.path.join(root, 'ixi_synth_data'))
        self.masks = load_h5(os.path.join(root, 'ixi_synth_masks'))
        if len(self.samples) > 0:
            print("Data loading successful. {} images collected from BRATS dataset.".format(len(self.samples)))
        else:
            print("Data loading error. Check the data path")
            sys.exit(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index].astype(np.float32)
        mask = self.masks[index].astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask


def get_loader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=False):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last
                      )
