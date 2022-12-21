from pystackreg import StackReg


def normalize(image):
    """
    Apply MinMax normalization.
    :param image: 2D numpy array representing a single image.
    :return image: normalized image
    """
    return (image - image.min()) / (image.max() - image.min())


def register(ref, mov, mode='rigid'):
    """
    Align image to the reference image
    :param ref: reference image, 2D numpy array
    :param mov: moved image, 2D numpy array of same shape as ref
    :param mode: type of transformation, options: (rigid, affine)
    :return: mov image registered to ref
    """
    if mode == 'rigid':
        sr = StackReg(StackReg.RIGID_BODY)
        img_reg = sr.register_transform(ref, mov)
    elif mode == 'affine':
        sr = StackReg(StackReg.AFFINE)
        img_reg = sr.register_transform(ref, mov)

    return img_reg

