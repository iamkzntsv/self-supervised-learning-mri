import glob


def load_paths(path):
    return [glob.glob(path + "/*.nii.gz", recursive=True)]
