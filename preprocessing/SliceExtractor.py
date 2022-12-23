import numpy as np
import tensorflow as tf
from deepbrain import Extractor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SliceExtractor:

    def __init__(self):
        self.ext = Extractor()

    def get_slices(self, volume):
        """
        Select 2D slices with relevant amounts of brain quantity from a 3D volume
        :param volume: nifti image representing MRI volume of a single subject
        :return: list of slices, where each slice is a numpy array of shape (256, 150)
        """
        nx, ny, nz = volume.header.get_data_shape()

        img_arr = self.reduce(volume.get_fdata())

        # Loop over axial plane
        slices = []
        for i in range(ny - 1):
            s = np.squeeze(img_arr[:, i:i + 1, :])
            s = np.pad(s, ((0, 0), (0, 150 - s.shape[1])))  # pad so that all slices have the same shape
            if np.sum(s) > 0:
                slices.append(s)

        return slices

    def reduce(self, img_arr, th=15000):
        """
        Compute brain quantity for each slice of the axial plane and set irrelevant slices to 0
        :param img_arr: numpy array of shape (nx, ny, nz)
        :param th: amount of brain quantity needed for the slice to be considered relevant
        :return:
        """
        # Skull stripping
        vox_mask = self.get_mask(img_arr) * 1
        img_arr_no_skull = img_arr * vox_mask

        # Compute brain quantity in axial plane (sum over coronal and sagittal axes)
        bq = np.apply_over_axes(np.sum, vox_mask, [0, 2])
        bq_mask = bq > th

        # Crop irrelevant parts of the volume
        img_arr_reduced = img_arr_no_skull * bq_mask
        return img_arr_reduced

    def get_mask(self, img, th=0.5):
        """
        Create a binary mask representing voxels with relevant amount of brain tissue
        :param img: image obtained after calling get_fdata() method, numpy array of shape (nx, ny, nz)
        :param th: amount of brain tissue need for a single voxel to be considered relevant
        :return: binary mask for the input image, numpy array of shape (nx, ny, nz)
        """
        # Get probability for each voxel being a brain tissue
        prob = self.ext.run(img)

        # Generate a mask
        mask = prob > th

        return mask


ext = SliceExtractor()
