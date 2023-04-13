import torch
import numpy as np
import h5py
import pickle
import wandb
import math


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)


def calc_activation_shape(dim, ksize=(5, 5), stride=(1, 1), padding=(0, 0), dilation=(1, 1), output_padding=(0, 0),
                          transposed=False):
    def shape_each_dim(i):
        if transposed:
            odim_i = (dim[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (ksize[i] - 1) + 1 + output_padding[i]
        else:
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            odim_i = odim_i / stride[i] + 1
        return math.floor(odim_i)

    return shape_each_dim(0), shape_each_dim(1)


def save_h5(data, name='dataset.h5'):
    hf = h5py.File(name, 'w')
    hf.create_dataset('data', data=data)
    hf.close()


def load_h5(name='dataset'):
    hf = h5py.File(name + '.h5', 'r')
    data = list(hf.get('data'))
    hf.close()
    return data


def load_artefact(dataset='ixi_dataset'):
    artifact = wandb.run.use_artifact(dataset + ':v0')
    data_dir = artifact.download()
    data = load_h5(data_dir + '\\' + dataset + '.h5')
    return data


def save_pickle(fname, file):
    with open(fname, 'wb') as f:
        pickle.dump(file, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def split_string(s):
    parts = s.rsplit('_', 1)
    return parts[0], parts[1]
