import nibabel as nib
import glob


def load_nifti(fname):
    volume = nib.load(fname)
    return volume


def load_paths(path):
    return [glob.glob(path + "/*.nii.gz", recursive=True)]


def get_average():
    # TO DO
    pass