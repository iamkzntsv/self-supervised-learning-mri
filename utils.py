import glob


def get_average_image():
    # TO DO
    pass


def load_paths(path):
    return [glob.glob(path + "/*.nii.gz", recursive=True)]
