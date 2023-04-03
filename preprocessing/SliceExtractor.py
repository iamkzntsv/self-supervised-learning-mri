import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os


class SliceExtractor:

    def __init__(self, bq_threshold=0.15, mq_threshold=0.03):
        """
        :param bq_threshold: Amount of brain quantity needed for a slice to be selected
        :param mq_threshold: Amount of abnormality needed for the brain slice to be selected
        """
        self.bq_threshold = bq_threshold
        self.mq_threshold = mq_threshold

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

                img = np.squeeze(img_arr[:, i:i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                mask = np.squeeze(mask_arr[:, i:i + 1, :])
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

                bq = self.compute_bq(img)
                if bq > self.bq_threshold:
                    img = self.center_crop(img, target_shape)
                    img_slices.append(img)

                    mask = self.center_crop(mask, target_shape)
                    mask_slices.append(mask)

            return img_slices, mask_slices
        else:
            img_slices = []
            for i in range(ny - 1):
                img = np.squeeze(img_arr[:, i:i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Select slices based on the amount of brain tissue
                bq = self.compute_bq(img)
                if bq > self.bq_threshold:
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
