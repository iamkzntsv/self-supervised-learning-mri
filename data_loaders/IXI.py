from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
from preprocessing.SliceExtractor import ext
from preprocessing.transforms import *
from utils import *


class IXI(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

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

        print("{} 2D slices collected from {} images.".format(len(images), len(fnames)))
        return images


def get_loader(root, batch_size, shuffle=True, num_workers=0, drop_last=False):
    transform = Compose([
        ToTensor(),
        Normalize(),
    ])

    dataset = IXI(root, transform)
    # reg_temp =

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=drop_last
                      )
