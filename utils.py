import nibabel as nib
import glob


def load_nifti(fname):
    img = nib.load(fname)
    return img


def load_paths(path):
    return [glob.glob(path + "/*.nii.gz", recursive=True)]


def get_average_image():
    # TO DO
    pass