import numpy as np
import cv2


class SliceExtractor:

    def __init__(self, bq_threshold=0.12, mq_threshold=0.03):
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

        target_shape = (224, 224)
        img_arr = self.center_crop(img_arr, target_shape)

        # Loop over axial plane
        if mask is not None:
            mask_arr = mask.get_fdata()
            mask_arr[mask_arr != 0] = 1
            mask_arr = self.center_crop(mask_arr, target_shape)

            img_slices = []
            mask_slices = []
            for i in range(nz - 1):
                img_2d = cv2.rotate(np.squeeze(img_arr[:, :, i:i + 1]), cv2.ROTATE_90_CLOCKWISE)
                mask_2d = cv2.rotate(np.squeeze(mask_arr[:, :, i:i + 1]), cv2.ROTATE_90_CLOCKWISE)

                # Relevant slice selection
                mq = self.compute_mq(mask_2d)
                if mq > self.mq_threshold:
                    img_slices.append(img_2d)
                    mask_slices.append(mask_2d)

            return img_slices, mask_slices
        else:
            img_slices = []
            for i in range(ny - 1):
                img = np.squeeze(img_arr[:, i:i + 1, :])
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = self.pad(img)

                # Select slices based on the amount of brain tissue
                bq = self.compute_bq(img)
                if bq > self.bq_threshold:
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
    def center_crop(arr, target_shape):
        start_indices = [(array_dim - target_dim) // 2 for array_dim, target_dim in zip(arr.shape, target_shape)]
        end_indices = [start + target_dim for start, target_dim in zip(start_indices, target_shape)]
        slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))

        return arr[slices]

    @staticmethod
    def pad(sample):
        padding = ((0, 0), (32, 32))
        return np.pad(sample, padding)


ext = SliceExtractor()
