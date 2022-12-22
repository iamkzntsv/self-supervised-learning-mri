import numpy as np
import tensorflow as tf
from deepbrain import Extractor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SliceExtractor:

    def __init__(self):
        self.ext = Extractor()

    def extract(self, img):
        """

        :param img:
        :return:
        """
        mask = self.get_mask(img)
        return img * mask

    def get_slices(self, vol):
        print()

    def get_mask(self, img, threshold=0.5):
        """
        Create a binary mask
        :param img: image obtained after calling get_fdata() method, numpy array of shape (250, 250, 156)
        :param threshold: threshold on the amount of brain tissue in a single voxel
        :return: binary mask for the input image, numpy array of shape (250, 250, 156)
        """
        # Get probability for each voxel being a brain tissue
        prob = self.ext.run(img)
        # Generate a mask
        mask = prob > threshold
        return mask


ext = SliceExtractor()

"""
def get_relevant_slices(self, img):
    nx, ny, nz = img.shape
    slices = []
    for i in range(ny - 1):
        slices.append(img[:, i:i + 1, :])

    return np.array(slices)
"""