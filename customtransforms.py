import torch
import random

class RandomHorizontalFlipTensor(object):
    """Horizontally flip the given Tensor Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def tesnor_hflip(self, tensor):
        """Flips tensor horizontally.
        """
        tensor = tensor.flip(2)
        return tensor


    def __call__(self, img):
        """
        Args:
            img (Tensor Image): Image to be flipped.

        Returns:
            Tensor Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self.tesnor_hflip(img)
        return img


class RandomVerticalFlipTensor(object):
    """Vertically flip the given Tensor Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def tesnor_vflip(self, tensor):
        """Flips tensor vertically.
        """
        tensor = tensor.flip(1)
        return tensor

    def __call__(self, img):
        """
        Args:
            img (Tensor Image): Image to be flipped.

        Returns:
            Tensor Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self.tesnor_vflip(img)
        return img