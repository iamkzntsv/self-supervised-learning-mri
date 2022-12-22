import nibabel as nib
import numpy as np
import tensorflow as tf
from deepbrain import Extractor
from matplotlib import pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SliceExtractor:

    def __init__(self):
        self.ext = Extractor()

    def get_slices(self, volume):
        """

        :param volume: Nifti1Image representing mri volume of a single subject
        :return:
        """
        nx, ny, nz = volume.header.get_data_shape()

        img_arr = volume.get_fdata()
        img_arr_cropped = self.crop(img_arr)

        """
        idx = 125
        plt.title("Slice {}, amount of brain quantity: {}".format(idx, np.squeeze(bq)[idx]))
        plt.imshow(np.squeeze(img_arr_cropped[:,idx:idx+1,:]), cmap='gray')
        plt.show()
        """

        exit()

        # Loop over axial plane
        slices = []
        for i in range(ny - 1):
            s = np.squeeze(img_arr[:,i:i+1,:])
            slices.append(s)

        return slices

    def crop(self, img_arr, th=8000):
        """
        Compute brain quantity for each slice of the axial plane and remove parts where the amount of tissue is not big enough
        :param img_arr: numpy array of shape (nx, ny, nz)
        :param th: threshold on the amount of brain quantity needed for the slice to be considered relevant
        :return:
        """
        # Perform skull stripping
        vox_mask = self.get_mask(img_arr) * 1
        img_arr_no_skull = img_arr * vox_mask

        # Compute brain quantity (sum over coronal and sagittal axes)
        bq = np.apply_over_axes(np.sum, vox_mask, [0, 2])
        bq_mask = bq > th

        # Crop irrelevant parts of the volume
        img_arr_cropped = img_arr_no_skull * bq_mask
        # TO DO: remove 0 slices

        return img_arr_cropped

    def get_mask(self, img, th=0.5):
        """
        Create a binary mask
        :param img: image obtained after calling get_fdata() method, numpy array of shape (nx, ny, nz)
        :param th: threshold on the amount of brain tissue in a single voxel
        :return: binary mask for the input image, numpy array of shape (nx, ny, nz)
        """
        # Get probability for each voxel being a brain tissue
        prob = self.ext.run(img)

        # Generate a mask
        mask = prob > th

        return mask


ext = SliceExtractor()
