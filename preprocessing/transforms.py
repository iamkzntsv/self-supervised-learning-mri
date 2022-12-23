from pystackreg import StackReg


class Normalize:
    """
    Apply MinMax normalization to a tensor.
    Args:
        sample (numpy array): Single image
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return (sample - sample.min()) / (sample.max() - sample.min())


class Register:
    """
    Align given image to the reference image
    Args:
        ref: reference image, numpy array
        mov moved image, numpy array of same shape as ref
        mode: type of transformation, options: (rigid, affine)
    """

    def __init__(self, mode='rigid'):
        self.mode = mode

    def __call__(self, ref, mov):
        if self.mode == 'rigid':
            sr = StackReg(StackReg.RIGID_BODY)
            img_reg = sr.register_transform(ref, mov)
        elif self.mode == 'affine':
            sr = StackReg(StackReg.AFFINE)
            img_reg = sr.register_transform(ref, mov)

        return img_reg
