import numpy as np
import cv2
from skimage import exposure


class SliceExtractor:

    def __init__(self, bq_threshold=0.15, mq_threshold=0.03):
        """
        :param bq_threshold: Amount of brain quantity needed for a slice to be selected
        :param mq_threshold: Amount of abnormality needed for the brain slice to be selected
        """
        self.bq_threshold = bq_threshold
        self.mq_threshold = mq_threshold
        self.hist_eq_reference = np.load('processing/ixi_reference_image.npy')

    def get_slices(self, volume, mask=None):
        """
        Extract 2D slices from a 3D volume based on the amounts of brain quantity
        :param volume: nifti image representing MRI volume of a single subject
        :param mask: nifti image representing segmentation mask for corresponding MRI volume
        :return: list of slices, where each slice is a numpy array of shape (256, 150)
        """
        nx, ny, nz = volume.header.get_data_shape()

        img_arr = volume.get_fdata()

        target_shape = (200, 200)

        # Loop over axial plane
        if mask is not None:
            mask_arr = mask.get_fdata()
            mask_arr[mask_arr != 0] = 1

            img_slices = []
            mask_slices = []

            for i in range(ny - 1):

                # Get slice, rotate
                img = np.squeeze(img_arr[:, i:i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                mask = np.squeeze(mask_arr[:, i:i + 1, :])
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

                mq = self.compute_mq(mask)
                if mq > self.mq_threshold:
                    # Normalization, equalization and cropping
                    img = img / np.max(img)
                    img = self.hist_equalize(img, self.hist_eq_reference)
                    img = self.center_crop(img, target_shape)
                    img_slices.append(img)

                    mask = self.center_crop(mask, target_shape)
                    mask_slices.append(mask)

            return img_slices, mask_slices
        else:
            img_slices = []
            for i in range(ny - 1):
                # Get slice, rotate and normalize
                img = np.squeeze(img_arr[:, i:i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Select slices based on the amount of brain tissue
                bq = self.compute_bq(img)
                if bq > self.bq_threshold:
                    # Normalize
                    img = img / np.max(img)
                    img = self.center_crop(img, target_shape)
                    img_slices.append(img)

            return img_slices

    @staticmethod
    def compute_bq(img):
        """
        Compute brain quantity
        :param img: a 2D brain slice
        :return:
        """
        return np.count_nonzero(img) / (img.shape[0] * img.shape[1])

    @staticmethod
    def compute_mq(mask):
        """
        Compute mask quantity
        :param mask: a 2D segmentation mask
        :return:
        """
        return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])

    @staticmethod
    def hist_equalize(source_img, reference_img):
        """
        Perform histogram equalization of source image to reference image
        :param source_img: source image
        :param reference_img: reference image
        :return: equalized source image
        """
        # Define non-black mask for reference image
        reference_mask = (reference_img > 0)

        # Define non-black mask for source image
        source_mask = (source_img > 0)

        # Perform histogram matching
        matched_image = exposure.match_histograms(source_img[source_mask], reference_img[reference_mask])

        # Create output image with non-black pixels replaced by matched pixels
        img_eq = np.zeros_like(source_img)
        img_eq[source_mask] = matched_image

        return img_eq

    @staticmethod
    def center_crop(img, size, offset=15):
        """
        Crops the center of the image to the specified size.
        :param img: A 2D numpy array.
        :param size: A tuple (width, height) specifying the size of the output image.
        :param offset: An integer specifying the vertical offset from the center of the image.

        :return: A numpy array containing the center-cropped image.
        """
        h, w = img.shape
        new_h, new_w = size

        top = int((h - new_h) / 2) + offset
        left = int((w - new_w) / 2)
        bottom = top + new_h
        right = left + new_w

        cropped_image = img[top:bottom, left:right]
        return cropped_image


ext = SliceExtractor()
