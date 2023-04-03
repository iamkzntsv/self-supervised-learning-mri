import h5py
import glob
import pickle
import wandb


def calc_activation_shape(dim, ksize=(5, 5), stride=(1, 1), padding=(0, 0)):
    import math

    def shape_each_dim(i):
        odim_i = dim[i] + 2 * padding[i] - (ksize[i] - 1) - 1
        return math.floor((odim_i / stride[i]) + 1)

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
