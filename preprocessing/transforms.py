from pystackreg import StackReg


class Normalize:
    """
    Apply MinMax normalization to a tensor.
    Args:
        sample (numpy array): Single image
    """
    def __call__(self, sample):
        return (sample - sample.min()) / (sample.max() - sample.min())


class Register:
    """
    Align sample to the reference image
    Args:
        ref: reference image, numpy array
        mov moved image, numpy array of same shape as ref
        mode: type of transformation, options: (rigid, affine)
    """
    def __call__(self, ref, mov, mode='rigid'):
        if mode == 'rigid':
            sr = StackReg(StackReg.RIGID_BODY)
            img_reg = sr.register_transform(ref, mov)
        elif mode == 'affine':
            sr = StackReg(StackReg.AFFINE)
            img_reg = sr.register_transform(ref, mov)

        else:
            raise NotImplementedError

        return img_reg
